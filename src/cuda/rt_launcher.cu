#include "rt_launcher.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "rt_kernels.cuh"

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)
#include <cuda.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "gafime_rt_optix_ptx.hpp"
#endif

namespace {

int cuda_status(cudaError_t status) {
    return status == cudaSuccess ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
}

bool decision_path_sign_supported(uint32_t sign) {
    return sign == GAFIME_DECISION_PATH_SIGN_LE || sign == GAFIME_DECISION_PATH_SIGN_GT;
}

bool rt_required(const GafimeDecisionPathBatch* batch) {
    return batch != nullptr && (batch->flags & GAFIME_DECISION_PATH_FLAG_REQUIRE_RT) != 0u;
}

int validate_decision_path_batch(
    const float* resident_features,
    uint64_t rows,
    uint32_t cols,
    const GafimeDecisionPathBatch* batch
) {
    if (resident_features == nullptr || batch == nullptr || batch->abi_version != GAFIME_ABI_VERSION) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if ((batch->flags & ~GAFIME_DECISION_PATH_FLAG_REQUIRE_RT) != 0u) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->path_count == 0 || batch->path_count == UINT32_MAX || batch->term_count == 0 ||
        batch->terms == nullptr || batch->path_offsets == nullptr || batch->membership_host == nullptr) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (batch->path_offsets[0] != 0) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t path_idx = 0; path_idx < batch->path_count; ++path_idx) {
        const uint32_t begin = batch->path_offsets[path_idx];
        const uint32_t end = batch->path_offsets[path_idx + 1];
        if (begin >= end || end > batch->term_count) {
            return GAFIME_STATUS_INVALID_ARGUMENT;
        }
        for (uint32_t term_idx = begin; term_idx < end; ++term_idx) {
            const GafimeDecisionPathTerm& term = batch->terms[term_idx];
            if (term.feature >= cols || !decision_path_sign_supported(term.sign)) {
                return GAFIME_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    if (batch->path_offsets[batch->path_count] != batch->term_count) {
        return GAFIME_STATUS_INVALID_ARGUMENT;
    }
    if (rows != 0 && batch->path_count > UINT64_MAX / rows) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    const uint64_t output_count = rows * static_cast<uint64_t>(batch->path_count);
    if (output_count > static_cast<uint64_t>(SIZE_MAX / sizeof(float))) {
        return GAFIME_STATUS_OUT_OF_MEMORY;
    }
    return GAFIME_STATUS_OK;
}

bool rt_disabled_by_env() {
    const char* mode = std::getenv("GAFIME_CUDA_DECISION_PATH_RT");
    return mode != nullptr && (mode[0] == '0' || mode[0] == 'n' || mode[0] == 'N' ||
        mode[0] == 'f' || mode[0] == 'F' || mode[0] == 'o' || mode[0] == 'O' ||
        mode[0] == 's' || mode[0] == 'S');
}

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)

bool cuda_arch_has_rt_cores(uint64_t arch_class) {
    return arch_class == GAFIME_GPU_ARCH_NVIDIA_TURING ||
        arch_class == GAFIME_GPU_ARCH_NVIDIA_AMPERE ||
        arch_class == GAFIME_GPU_ARCH_NVIDIA_ADA ||
        arch_class == GAFIME_GPU_ARCH_NVIDIA_HOPPER ||
        arch_class == GAFIME_GPU_ARCH_NVIDIA_BLACKWELL;
}

bool append_unique_axis(std::vector<uint32_t>& axes, uint32_t feature) {
    if (std::find(axes.begin(), axes.end(), feature) != axes.end()) {
        return true;
    }
    if (axes.size() >= 3) {
        return false;
    }
    axes.push_back(feature);
    return true;
}

uint32_t axis_index(const std::array<uint32_t, 3>& axes, uint32_t dims, uint32_t feature) {
    for (uint32_t idx = 0; idx < dims; ++idx) {
        if (axes[idx] == feature) {
            return idx;
        }
    }
    return UINT32_MAX;
}

struct RtBoxPlan {
    std::array<uint32_t, 3> axes{0, 0, 0};
    uint32_t dims = 0;
    std::vector<gafime_cuda_v1::rt_kernel::GafimeRtBox> boxes;
};

int build_rt_box_plan(const GafimeDecisionPathBatch* paths, RtBoxPlan& plan) {
    std::vector<uint32_t> axes;
    axes.reserve(3);
    for (uint32_t term_idx = 0; term_idx < paths->term_count; ++term_idx) {
        const GafimeDecisionPathTerm& term = paths->terms[term_idx];
        if (!std::isfinite(term.threshold)) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
        if (!append_unique_axis(axes, term.feature)) {
            return GAFIME_STATUS_UNSUPPORTED_BACKEND;
        }
    }
    std::sort(axes.begin(), axes.end());
    plan.dims = static_cast<uint32_t>(axes.size());
    if (plan.dims == 0 || plan.dims > 3) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    for (uint32_t idx = 0; idx < plan.dims; ++idx) {
        plan.axes[idx] = axes[idx];
    }

    plan.boxes.assign(paths->path_count, {});
    for (uint32_t path_idx = 0; path_idx < paths->path_count; ++path_idx) {
        float lo[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        float hi[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
        uint32_t open_lo_mask = 0;
        const uint32_t begin = paths->path_offsets[path_idx];
        const uint32_t end = paths->path_offsets[path_idx + 1];
        for (uint32_t term_idx = begin; term_idx < end; ++term_idx) {
            const GafimeDecisionPathTerm& term = paths->terms[term_idx];
            const uint32_t axis = axis_index(plan.axes, plan.dims, term.feature);
            if (axis == UINT32_MAX) {
                return GAFIME_STATUS_UNSUPPORTED_BACKEND;
            }
            if (term.sign == GAFIME_DECISION_PATH_SIGN_LE) {
                hi[axis] = std::min(hi[axis], term.threshold);
            } else {
                if (term.threshold >= lo[axis]) {
                    lo[axis] = term.threshold;
                    open_lo_mask |= (1u << axis);
                }
            }
        }
        for (uint32_t axis = 0; axis < plan.dims; ++axis) {
            if (lo[axis] > hi[axis] || (lo[axis] == hi[axis] && (open_lo_mask & (1u << axis)) != 0u)) {
                lo[axis] = 0.0f;
                hi[axis] = 0.0f;
                open_lo_mask |= (1u << axis);
            }
        }
        plan.boxes[path_idx] = {
            lo[0],
            lo[1],
            lo[2],
            hi[0],
            hi[1],
            hi[2],
            open_lo_mask,
            plan.dims,
        };
    }
    return GAFIME_STATUS_OK;
}

#endif

int execute_decision_path_membership_sm(
    const float* resident_features,
    uint64_t rows,
    uint32_t cols,
    const GafimeDecisionPathBatch* paths
) {
    const uint64_t output_count = rows * static_cast<uint64_t>(paths->path_count);
    const size_t term_bytes = static_cast<size_t>(paths->term_count) * sizeof(GafimeDecisionPathTerm);
    const size_t offset_bytes = static_cast<size_t>(paths->path_count + 1u) * sizeof(uint32_t);
    const size_t output_bytes = static_cast<size_t>(output_count) * sizeof(float);

    GafimeDecisionPathTerm* terms_device = nullptr;
    uint32_t* offsets_device = nullptr;
    float* membership_device = nullptr;

    int status = cuda_status(cudaMalloc(&terms_device, term_bytes));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&offsets_device, offset_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&membership_device, output_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(terms_device, paths->terms, term_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(offsets_device, paths->path_offsets, offset_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(gafime_cuda_v1::launch_decision_path_membership(
            resident_features,
            rows,
            cols,
            terms_device,
            offsets_device,
            paths->path_count,
            membership_device,
            0
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaDeviceSynchronize());
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(paths->membership_host, membership_device, output_bytes, cudaMemcpyDeviceToHost));
    }

    cudaFree(membership_device);
    cudaFree(offsets_device);
    cudaFree(terms_device);
    return status;
}

#if defined(GAFIME_CUDA_ENABLE_OPTIX_RT)

struct GafimeRtParams {
    OptixTraversableHandle handle;
    const float* points_xyz;
    const gafime_cuda_v1::rt_kernel::GafimeRtBox* boxes;
    float* membership;
    uint32_t rows;
    uint32_t path_count;
};

struct EmptySbtData {};

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using EmptyRecord = SbtRecord<EmptySbtData>;

int optix_status(OptixResult status) {
    return status == OPTIX_SUCCESS ? GAFIME_STATUS_OK : GAFIME_STATUS_DEVICE_ERROR;
}

struct RtOptixProgram {
    uint32_t device_id = UINT32_MAX;
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup program_groups[3]{};
    OptixPipeline pipeline = nullptr;
    EmptyRecord* raygen_record = nullptr;
    EmptyRecord* miss_record = nullptr;
    EmptyRecord* hitgroup_record = nullptr;
    OptixShaderBindingTable sbt{};

    ~RtOptixProgram() = default;

    void reset() {
        cudaFree(hitgroup_record);
        cudaFree(miss_record);
        cudaFree(raygen_record);
        hitgroup_record = nullptr;
        miss_record = nullptr;
        raygen_record = nullptr;
        if (pipeline != nullptr) {
            optixPipelineDestroy(pipeline);
            pipeline = nullptr;
        }
        for (OptixProgramGroup& program_group : program_groups) {
            if (program_group != nullptr) {
                optixProgramGroupDestroy(program_group);
                program_group = nullptr;
            }
        }
        if (module != nullptr) {
            optixModuleDestroy(module);
            module = nullptr;
        }
        if (context != nullptr) {
            optixDeviceContextDestroy(context);
            context = nullptr;
        }
        sbt = {};
        device_id = UINT32_MAX;
    }

    bool ready(uint32_t wanted_device_id) const {
        return context != nullptr && pipeline != nullptr && device_id == wanted_device_id;
    }
};

RtOptixProgram& optix_program() {
    static RtOptixProgram program;
    return program;
}

int ensure_optix_program(uint32_t device_id) {
    RtOptixProgram& program = optix_program();
    if (program.ready(device_id)) {
        return GAFIME_STATUS_OK;
    }
    program.reset();

    if (cudaFree(nullptr) != cudaSuccess || optixInit() != OPTIX_SUCCESS) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    CUcontext cu_ctx = nullptr;
    if (cuCtxGetCurrent(&cu_ctx) != CUDA_SUCCESS) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    OptixDeviceContextOptions context_options = {};
    int status = optix_status(optixDeviceContextCreate(cu_ctx, &context_options, &program.context));
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = false;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 1;
    pipeline_options.numAttributeValues = 0;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    char log[4096];
    size_t log_size = sizeof(log);
    status = optix_status(optixModuleCreate(
        program.context,
        &module_options,
        &pipeline_options,
        gafime_cuda_v1::kRtOptixPtx,
        gafime_cuda_v1::kRtOptixPtxSize,
        log,
        &log_size,
        &program.module
    ));
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return status;
    }

    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc pg_descs[3] = {};
    pg_descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pg_descs[0].raygen.module = program.module;
    pg_descs[0].raygen.entryFunctionName = "__raygen__gafime_dp";
    pg_descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pg_descs[1].miss.module = program.module;
    pg_descs[1].miss.entryFunctionName = "__miss__gafime_dp";
    pg_descs[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_descs[2].hitgroup.moduleIS = program.module;
    pg_descs[2].hitgroup.entryFunctionNameIS = "__intersection__gafime_dp_box";
    pg_descs[2].hitgroup.moduleAH = program.module;
    pg_descs[2].hitgroup.entryFunctionNameAH = "__anyhit__gafime_dp_mark";

    log_size = sizeof(log);
    status = optix_status(optixProgramGroupCreate(
        program.context,
        pg_descs,
        3,
        &pg_options,
        log,
        &log_size,
        program.program_groups
    ));
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return status;
    }

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    status = optix_status(optixPipelineCreate(
        program.context,
        &pipeline_options,
        &link_options,
        program.program_groups,
        3,
        log,
        &log_size,
        &program.pipeline
    ));
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixPipelineSetStackSize(program.pipeline, 0, 0, 0, 1));
    }
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return status;
    }

    EmptyRecord raygen_record = {};
    EmptyRecord miss_record = {};
    EmptyRecord hitgroup_record = {};
    status = optix_status(optixSbtRecordPackHeader(program.program_groups[0], &raygen_record));
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixSbtRecordPackHeader(program.program_groups[1], &miss_record));
    }
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixSbtRecordPackHeader(program.program_groups[2], &hitgroup_record));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&program.raygen_record, sizeof(EmptyRecord)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&program.miss_record, sizeof(EmptyRecord)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&program.hitgroup_record, sizeof(EmptyRecord)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(program.raygen_record, &raygen_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(program.miss_record, &miss_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(program.hitgroup_record, &hitgroup_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    }
    if (status != GAFIME_STATUS_OK) {
        program.reset();
        return status;
    }

    program.sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(program.raygen_record);
    program.sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(program.miss_record);
    program.sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    program.sbt.missRecordCount = 1;
    program.sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(program.hitgroup_record);
    program.sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    program.sbt.hitgroupRecordCount = 1;
    program.device_id = device_id;
    return GAFIME_STATUS_OK;
}

int execute_decision_path_membership_optix(
    const float* resident_features,
    uint64_t rows,
    uint32_t device_id,
    uint64_t arch_class,
    bool features_are_finite,
    const GafimeDecisionPathBatch* paths
) {
    if (!features_are_finite || !cuda_arch_has_rt_cores(arch_class)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }
    if (rows > UINT32_MAX) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    RtBoxPlan plan;
    int status = build_rt_box_plan(paths, plan);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }
    status = ensure_optix_program(device_id);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    const uint64_t output_count = rows * static_cast<uint64_t>(paths->path_count);
    const size_t points_bytes = static_cast<size_t>(rows) * 3u * sizeof(float);
    const size_t box_bytes = static_cast<size_t>(paths->path_count) * sizeof(gafime_cuda_v1::rt_kernel::GafimeRtBox);
    const size_t output_bytes = static_cast<size_t>(output_count) * sizeof(float);

    std::vector<OptixAabb> aabbs;
    aabbs.reserve(plan.boxes.size());
    for (const gafime_cuda_v1::rt_kernel::GafimeRtBox& box : plan.boxes) {
        aabbs.push_back({box.lo_x, box.lo_y, box.lo_z, box.hi_x, box.hi_y, box.hi_z});
    }

    float* points_device = nullptr;
    gafime_cuda_v1::rt_kernel::GafimeRtBox* boxes_device = nullptr;
    float* membership_device = nullptr;
    OptixAabb* aabbs_device = nullptr;
    void* gas_temp_device = nullptr;
    void* gas_output_device = nullptr;
    GafimeRtParams* params_device = nullptr;
    cudaStream_t stream = nullptr;

    status = cuda_status(cudaMalloc(&points_device, points_bytes));
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&boxes_device, box_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&membership_device, output_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&aabbs_device, aabbs.size() * sizeof(OptixAabb)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(boxes_device, plan.boxes.data(), box_bytes, cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(aabbs_device, aabbs.data(), aabbs.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemset(membership_device, 0, output_bytes));
    }
    if (status == GAFIME_STATUS_OK) {
        constexpr uint32_t threads = 256;
        const uint32_t row_blocks = static_cast<uint32_t>((rows + threads - 1) / threads);
        gafime_cuda_v1::rt_kernel::pack_decision_path_points_kernel<<<row_blocks, threads>>>(
            resident_features,
            rows,
            plan.axes[0],
            plan.axes[1],
            plan.axes[2],
            plan.dims,
            points_device
        );
        status = cuda_status(cudaGetLastError());
    }

    CUdeviceptr aabb_buffer = reinterpret_cast<CUdeviceptr>(aabbs_device);
    uint32_t geometry_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
    build_input.customPrimitiveArray.numPrimitives = paths->path_count;
    build_input.customPrimitiveArray.flags = geometry_flags;
    build_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes gas_sizes = {};
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixAccelComputeMemoryUsage(
            optix_program().context,
            &accel_options,
            &build_input,
            1,
            &gas_sizes
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&gas_temp_device, gas_sizes.tempSizeInBytes));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&gas_output_device, gas_sizes.outputSizeInBytes));
    }
    OptixTraversableHandle gas_handle = 0;
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixAccelBuild(
            optix_program().context,
            0,
            &accel_options,
            &build_input,
            1,
            reinterpret_cast<CUdeviceptr>(gas_temp_device),
            gas_sizes.tempSizeInBytes,
            reinterpret_cast<CUdeviceptr>(gas_output_device),
            gas_sizes.outputSizeInBytes,
            &gas_handle,
            nullptr,
            0
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaDeviceSynchronize());
    }

    GafimeRtParams params = {};
    params.handle = gas_handle;
    params.points_xyz = points_device;
    params.boxes = boxes_device;
    params.membership = membership_device;
    params.rows = static_cast<uint32_t>(rows);
    params.path_count = paths->path_count;
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMalloc(&params_device, sizeof(GafimeRtParams)));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(params_device, &params, sizeof(params), cudaMemcpyHostToDevice));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaStreamCreate(&stream));
    }
    if (status == GAFIME_STATUS_OK) {
        status = optix_status(optixLaunch(
            optix_program().pipeline,
            stream,
            reinterpret_cast<CUdeviceptr>(params_device),
            sizeof(GafimeRtParams),
            &optix_program().sbt,
            static_cast<uint32_t>(rows),
            1,
            1
        ));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaStreamSynchronize(stream));
    }
    if (status == GAFIME_STATUS_OK) {
        status = cuda_status(cudaMemcpy(paths->membership_host, membership_device, output_bytes, cudaMemcpyDeviceToHost));
    }

    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    cudaFree(params_device);
    cudaFree(gas_output_device);
    cudaFree(gas_temp_device);
    cudaFree(aabbs_device);
    cudaFree(membership_device);
    cudaFree(boxes_device);
    cudaFree(points_device);
    return status;
}

#else

int execute_decision_path_membership_optix(
    const float*,
    uint64_t,
    uint32_t,
    uint64_t,
    bool,
    const GafimeDecisionPathBatch*
) {
    return GAFIME_STATUS_UNSUPPORTED_BACKEND;
}

#endif

}  // namespace

namespace gafime_cuda_v1 {

void tune_rt_kernels_for_device(const cudaDeviceProp& props) {
    const cudaFuncCache cache_mode = props.major >= 7 ? cudaFuncCachePreferShared : cudaFuncCachePreferL1;
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::rt_kernel::decision_path_membership_kernel,
        cache_mode
    ));
    static_cast<void>(cudaFuncSetCacheConfig(
        gafime_cuda_v1::rt_kernel::pack_decision_path_points_kernel,
        cache_mode
    ));
}

cudaError_t launch_decision_path_membership(
    const float* features,
    uint64_t n_samples,
    uint32_t n_features,
    const GafimeDecisionPathTerm* terms,
    const uint32_t* path_offsets,
    uint32_t path_count,
    float* membership,
    cudaStream_t stream
) {
    if (path_count == 0 || n_samples == 0) {
        return cudaSuccess;
    }
    constexpr uint32_t threads = 256;
    const uint32_t row_blocks = static_cast<uint32_t>((n_samples + threads - 1) / threads);
    dim3 grid(path_count, row_blocks);
    dim3 block(threads);
    rt_kernel::decision_path_membership_kernel<<<grid, block, 0, stream>>>(
        features,
        n_samples,
        n_features,
        terms,
        path_offsets,
        path_count,
        membership
    );
    return cudaGetLastError();
}

int execute_decision_path_membership(
    const float* resident_features,
    uint64_t rows,
    uint32_t cols,
    uint32_t device_id,
    uint64_t arch_class,
    uint32_t device_flags,
    bool features_are_finite,
    const GafimeDecisionPathBatch* paths
) {
    static_cast<void>(device_flags);
    int status = validate_decision_path_batch(resident_features, rows, cols, paths);
    if (status != GAFIME_STATUS_OK) {
        return status;
    }

    if (!rt_disabled_by_env()) {
        status = execute_decision_path_membership_optix(
            resident_features,
            rows,
            device_id,
            arch_class,
            features_are_finite,
            paths
        );
        if (status == GAFIME_STATUS_OK) {
            return status;
        }
        if (status != GAFIME_STATUS_UNSUPPORTED_BACKEND || rt_required(paths)) {
            return status;
        }
    } else if (rt_required(paths)) {
        return GAFIME_STATUS_UNSUPPORTED_BACKEND;
    }

    return execute_decision_path_membership_sm(resident_features, rows, cols, paths);
}

}  // namespace gafime_cuda_v1
