#include <cstdint>

#ifdef GAFIME_OPTIX_DEVICE

#include <optix.h>
#include <optix_device.h>

struct GafimeRtBox {
    float lo_x;
    float lo_y;
    float lo_z;
    float hi_x;
    float hi_y;
    float hi_z;
    uint32_t open_lo_mask;
    uint32_t dims;
};

struct GafimeRtParams {
    OptixTraversableHandle handle;
    const float* points_xyz;
    const GafimeRtBox* boxes;
    float* membership;
    uint32_t rows;
    uint32_t path_count;
};

extern "C" {
__constant__ GafimeRtParams params;
}

static __forceinline__ __device__ bool inside_dim(
    float value,
    float lo,
    float hi,
    bool lo_open
) {
    return lo_open ? (value > lo && value <= hi) : (value >= lo && value <= hi);
}

extern "C" __global__ void __raygen__gafime_dp()
{
    const uint32_t row = optixGetLaunchIndex().x;
    if (row >= params.rows) {
        return;
    }

    const float x = params.points_xyz[row * 3u + 0u];
    const float y = params.points_xyz[row * 3u + 1u];
    const float z = params.points_xyz[row * 3u + 2u];
    uint32_t payload_row = row;

    optixTrace(
        params.handle,
        make_float3(x, y, z),
        make_float3(1.0f, 0.0f, 0.0f),
        0.0f,
        1.0e-7f,
        0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0,
        1,
        0,
        payload_row
    );
}

extern "C" __global__ void __miss__gafime_dp() {}

extern "C" __global__ void __intersection__gafime_dp_box()
{
    const uint32_t primitive_idx = optixGetPrimitiveIndex();
    const float3 point = optixGetWorldRayOrigin();
    const GafimeRtBox box = params.boxes[primitive_idx];

    bool inside = inside_dim(point.x, box.lo_x, box.hi_x, (box.open_lo_mask & 1u) != 0u);
    if (box.dims > 1u) {
        inside = inside && inside_dim(point.y, box.lo_y, box.hi_y, (box.open_lo_mask & 2u) != 0u);
    }
    if (box.dims > 2u) {
        inside = inside && inside_dim(point.z, box.lo_z, box.hi_z, (box.open_lo_mask & 4u) != 0u);
    }
    if (inside) {
        optixReportIntersection(0.0f, 0);
    }
}

extern "C" __global__ void __anyhit__gafime_dp_mark()
{
    const uint32_t row = optixGetPayload_0();
    const uint32_t primitive_idx = optixGetPrimitiveIndex();
    params.membership[static_cast<uint64_t>(primitive_idx) * params.rows + row] = 1.0f;
    optixIgnoreIntersection();
}

#else

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct GafimeRtBox {
    float lo_x;
    float lo_y;
    float lo_z;
    float hi_x;
    float hi_y;
    float hi_z;
    uint32_t open_lo_mask;
    uint32_t dims;
};

struct GafimeRtParams {
    OptixTraversableHandle handle;
    const float* points_xyz;
    const GafimeRtBox* boxes;
    float* membership;
    uint32_t rows;
    uint32_t path_count;
};

template <typename T>
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptySbtData {};

__global__ void sm_decision_path_membership_kernel(
    const float* points_xyz,
    const GafimeRtBox* boxes,
    float* membership,
    uint32_t rows,
    uint32_t path_count
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t path = blockIdx.y;
    if (row >= rows || path >= path_count) {
        return;
    }
    const GafimeRtBox box = boxes[path];
    const float values[3] = {
        points_xyz[row * 3u + 0u],
        points_xyz[row * 3u + 1u],
        points_xyz[row * 3u + 2u],
    };
    bool inside = true;
    const float lo[3] = {box.lo_x, box.lo_y, box.lo_z};
    const float hi[3] = {box.hi_x, box.hi_y, box.hi_z};
    for (uint32_t dim = 0; dim < box.dims; ++dim) {
        const bool lo_open = (box.open_lo_mask & (1u << dim)) != 0u;
        inside = inside && (lo_open ? values[dim] > lo[dim] : values[dim] >= lo[dim]);
        inside = inside && values[dim] <= hi[dim];
    }
    membership[static_cast<uint64_t>(path) * rows + row] = inside ? 1.0f : 0.0f;
}

void check_cuda(cudaError_t status, const char* label)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

void check_optix(OptixResult status, const char* label)
{
    if (status != OPTIX_SUCCESS) {
        throw std::runtime_error(std::string(label) + ": OptiX status " + std::to_string(status));
    }
}

std::string read_file(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error(std::string("failed to open PTX: ") + path);
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

float time_sm(
    const float* d_points,
    const GafimeRtBox* d_boxes,
    float* d_membership,
    uint32_t rows,
    uint32_t path_count
) {
    check_cuda(cudaMemset(d_membership, 0, static_cast<size_t>(rows) * path_count * sizeof(float)), "sm memset");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "sm event create start");
    check_cuda(cudaEventCreate(&stop), "sm event create stop");
    dim3 block(256);
    dim3 grid((rows + block.x - 1u) / block.x, path_count);
    check_cuda(cudaEventRecord(start), "sm event start");
    sm_decision_path_membership_kernel<<<grid, block>>>(d_points, d_boxes, d_membership, rows, path_count);
    check_cuda(cudaGetLastError(), "sm launch");
    check_cuda(cudaEventRecord(stop), "sm event stop");
    check_cuda(cudaEventSynchronize(stop), "sm sync");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "sm elapsed");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

float time_optix(
    OptixPipeline pipeline,
    CUstream stream,
    CUdeviceptr d_params,
    const OptixShaderBindingTable* sbt,
    float* d_membership,
    uint32_t rows,
    uint32_t path_count
) {
    check_cuda(cudaMemset(d_membership, 0, static_cast<size_t>(rows) * path_count * sizeof(float)), "optix memset");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "optix event create start");
    check_cuda(cudaEventCreate(&stop), "optix event create stop");
    check_cuda(cudaEventRecord(start, stream), "optix event start");
    check_optix(optixLaunch(pipeline, stream, d_params, sizeof(GafimeRtParams), sbt, rows, 1, 1), "optix launch");
    check_cuda(cudaEventRecord(stop, stream), "optix event stop");
    check_cuda(cudaEventSynchronize(stop), "optix sync");
    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "optix elapsed");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return ms;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " gafime_rt_decision_path_optix.ptx\n";
        return 2;
    }

    try {
        const std::string ptx = read_file(argv[1]);
        check_cuda(cudaFree(nullptr), "cuda init");
        check_optix(optixInit(), "optix init");

        CUcontext cu_ctx = nullptr;
        OptixDeviceContext context = nullptr;
        OptixDeviceContextOptions context_options = {};
        check_optix(optixDeviceContextCreate(cu_ctx, &context_options, &context), "context create");

        constexpr uint32_t rows = 4;
        constexpr uint32_t path_count = 2;
        const std::vector<float> points = {
            1.0f, 5.0f, 0.0f,
            2.0f, 4.0f, 0.0f,
            3.0f, 3.0f, 0.0f,
            4.0f, 2.0f, 0.0f,
        };
        const std::vector<GafimeRtBox> boxes = {
            {-1.0e20f, -1.0e20f, -1.0e20f, 2.0f, 1.0e20f, 1.0e20f, 0u, 1u},
            {2.0f, 2.0f, -1.0e20f, 1.0e20f, 1.0e20f, 1.0e20f, 3u, 2u},
        };
        const std::vector<float> expected = {
            1.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
        };

        float* d_points = nullptr;
        GafimeRtBox* d_boxes = nullptr;
        float* d_sm_membership = nullptr;
        float* d_optix_membership = nullptr;
        check_cuda(cudaMalloc(&d_points, points.size() * sizeof(float)), "points malloc");
        check_cuda(cudaMalloc(&d_boxes, boxes.size() * sizeof(GafimeRtBox)), "boxes malloc");
        check_cuda(cudaMalloc(&d_sm_membership, expected.size() * sizeof(float)), "sm membership malloc");
        check_cuda(cudaMalloc(&d_optix_membership, expected.size() * sizeof(float)), "optix membership malloc");
        check_cuda(cudaMemcpy(d_points, points.data(), points.size() * sizeof(float), cudaMemcpyHostToDevice), "points copy");
        check_cuda(cudaMemcpy(d_boxes, boxes.data(), boxes.size() * sizeof(GafimeRtBox), cudaMemcpyHostToDevice), "boxes copy");

        OptixAabb* d_aabbs = nullptr;
        std::vector<OptixAabb> aabbs;
        aabbs.reserve(boxes.size());
        for (const GafimeRtBox& box : boxes) {
            aabbs.push_back({box.lo_x, box.lo_y, box.lo_z, box.hi_x, box.hi_y, box.hi_z});
        }
        check_cuda(cudaMalloc(&d_aabbs, aabbs.size() * sizeof(OptixAabb)), "aabb malloc");
        check_cuda(cudaMemcpy(d_aabbs, aabbs.data(), aabbs.size() * sizeof(OptixAabb), cudaMemcpyHostToDevice), "aabb copy");

        CUdeviceptr d_aabb_buffer = reinterpret_cast<CUdeviceptr>(d_aabbs);
        uint32_t geometry_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
        build_input.customPrimitiveArray.numPrimitives = path_count;
        build_input.customPrimitiveArray.flags = geometry_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes gas_sizes = {};
        check_optix(
            optixAccelComputeMemoryUsage(context, &accel_options, &build_input, 1, &gas_sizes),
            "gas memory usage"
        );
        void* d_gas_temp = nullptr;
        void* d_gas_output = nullptr;
        check_cuda(cudaMalloc(&d_gas_temp, gas_sizes.tempSizeInBytes), "gas temp malloc");
        check_cuda(cudaMalloc(&d_gas_output, gas_sizes.outputSizeInBytes), "gas output malloc");
        OptixTraversableHandle gas_handle = 0;
        check_optix(
            optixAccelBuild(
                context,
                0,
                &accel_options,
                &build_input,
                1,
                reinterpret_cast<CUdeviceptr>(d_gas_temp),
                gas_sizes.tempSizeInBytes,
                reinterpret_cast<CUdeviceptr>(d_gas_output),
                gas_sizes.outputSizeInBytes,
                &gas_handle,
                nullptr,
                0
            ),
            "gas build"
        );
        check_cuda(cudaDeviceSynchronize(), "gas sync");

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
        OptixModule module = nullptr;
        check_optix(
            optixModuleCreate(
                context,
                &module_options,
                &pipeline_options,
                ptx.c_str(),
                ptx.size(),
                log,
                &log_size,
                &module
            ),
            "module create"
        );

        OptixProgramGroupOptions pg_options = {};
        OptixProgramGroupDesc pg_descs[3] = {};
        pg_descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pg_descs[0].raygen.module = module;
        pg_descs[0].raygen.entryFunctionName = "__raygen__gafime_dp";
        pg_descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pg_descs[1].miss.module = module;
        pg_descs[1].miss.entryFunctionName = "__miss__gafime_dp";
        pg_descs[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pg_descs[2].hitgroup.moduleIS = module;
        pg_descs[2].hitgroup.entryFunctionNameIS = "__intersection__gafime_dp_box";
        pg_descs[2].hitgroup.moduleAH = module;
        pg_descs[2].hitgroup.entryFunctionNameAH = "__anyhit__gafime_dp_mark";

        OptixProgramGroup program_groups[3] = {};
        log_size = sizeof(log);
        check_optix(
            optixProgramGroupCreate(context, pg_descs, 3, &pg_options, log, &log_size, program_groups),
            "program groups"
        );

        OptixPipelineLinkOptions link_options = {};
        link_options.maxTraceDepth = 1;
        OptixPipeline pipeline = nullptr;
        log_size = sizeof(log);
        check_optix(
            optixPipelineCreate(
                context,
                &pipeline_options,
                &link_options,
                program_groups,
                3,
                log,
                &log_size,
                &pipeline
            ),
            "pipeline create"
        );
        check_optix(optixPipelineSetStackSize(pipeline, 0, 0, 0, 1), "pipeline stack");

        using EmptyRecord = SbtRecord<EmptySbtData>;
        EmptyRecord raygen_record = {};
        EmptyRecord miss_record = {};
        EmptyRecord hitgroup_record = {};
        check_optix(optixSbtRecordPackHeader(program_groups[0], &raygen_record), "raygen sbt pack");
        check_optix(optixSbtRecordPackHeader(program_groups[1], &miss_record), "miss sbt pack");
        check_optix(optixSbtRecordPackHeader(program_groups[2], &hitgroup_record), "hitgroup sbt pack");
        EmptyRecord* d_raygen_record = nullptr;
        EmptyRecord* d_miss_record = nullptr;
        EmptyRecord* d_hitgroup_record = nullptr;
        check_cuda(cudaMalloc(&d_raygen_record, sizeof(EmptyRecord)), "raygen sbt malloc");
        check_cuda(cudaMalloc(&d_miss_record, sizeof(EmptyRecord)), "miss sbt malloc");
        check_cuda(cudaMalloc(&d_hitgroup_record, sizeof(EmptyRecord)), "hitgroup sbt malloc");
        check_cuda(cudaMemcpy(d_raygen_record, &raygen_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice), "raygen sbt copy");
        check_cuda(cudaMemcpy(d_miss_record, &miss_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice), "miss sbt copy");
        check_cuda(cudaMemcpy(d_hitgroup_record, &hitgroup_record, sizeof(EmptyRecord), cudaMemcpyHostToDevice), "hitgroup sbt copy");

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen_record);
        sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(d_miss_record);
        sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(d_hitgroup_record);
        sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
        sbt.hitgroupRecordCount = 1;

        GafimeRtParams params = {};
        params.handle = gas_handle;
        params.points_xyz = d_points;
        params.boxes = d_boxes;
        params.membership = d_optix_membership;
        params.rows = rows;
        params.path_count = path_count;
        GafimeRtParams* d_params_raw = nullptr;
        check_cuda(cudaMalloc(&d_params_raw, sizeof(GafimeRtParams)), "params malloc");
        check_cuda(cudaMemcpy(d_params_raw, &params, sizeof(params), cudaMemcpyHostToDevice), "params copy");

        CUstream stream = nullptr;
        check_cuda(cudaStreamCreate(&stream), "stream create");
        static_cast<void>(time_sm(d_points, d_boxes, d_sm_membership, rows, path_count));
        static_cast<void>(time_optix(
            pipeline,
            stream,
            reinterpret_cast<CUdeviceptr>(d_params_raw),
            &sbt,
            d_optix_membership,
            rows,
            path_count
        ));
        const float sm_ms = time_sm(d_points, d_boxes, d_sm_membership, rows, path_count);
        const float optix_ms = time_optix(
            pipeline,
            stream,
            reinterpret_cast<CUdeviceptr>(d_params_raw),
            &sbt,
            d_optix_membership,
            rows,
            path_count
        );

        std::vector<float> sm(expected.size(), -1.0f);
        std::vector<float> optix(expected.size(), -1.0f);
        check_cuda(cudaMemcpy(sm.data(), d_sm_membership, sm.size() * sizeof(float), cudaMemcpyDeviceToHost), "sm copy out");
        check_cuda(cudaMemcpy(optix.data(), d_optix_membership, optix.size() * sizeof(float), cudaMemcpyDeviceToHost), "optix copy out");
        for (size_t idx = 0; idx < expected.size(); ++idx) {
            if (sm[idx] != expected[idx] || optix[idx] != expected[idx]) {
                std::cerr << "membership mismatch at " << idx
                          << " expected=" << expected[idx]
                          << " sm=" << sm[idx]
                          << " optix=" << optix[idx] << "\n";
                return 1;
            }
        }

        std::cout << "optix decision_path AABB parity passed; sm_ms=" << sm_ms
                  << " optix_ms=" << optix_ms << "\n";

        cudaStreamDestroy(stream);
        cudaFree(d_params_raw);
        cudaFree(d_hitgroup_record);
        cudaFree(d_miss_record);
        cudaFree(d_raygen_record);
        optixPipelineDestroy(pipeline);
        optixProgramGroupDestroy(program_groups[2]);
        optixProgramGroupDestroy(program_groups[1]);
        optixProgramGroupDestroy(program_groups[0]);
        optixModuleDestroy(module);
        cudaFree(d_gas_output);
        cudaFree(d_gas_temp);
        cudaFree(d_aabbs);
        cudaFree(d_optix_membership);
        cudaFree(d_sm_membership);
        cudaFree(d_boxes);
        cudaFree(d_points);
        optixDeviceContextDestroy(context);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n";
        return 1;
    }
    return 0;
}

#endif
