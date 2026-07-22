use gafime_types::{
    BackendKind, GafimeGpuDeviceInfo, GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
    GAFIME_GPU_ARCH_AMD_CDNA, GAFIME_GPU_ARCH_AMD_RDNA, GAFIME_GPU_ARCH_APPLE,
    GAFIME_GPU_ARCH_NVIDIA_ADA, GAFIME_GPU_ARCH_NVIDIA_AMPERE, GAFIME_GPU_ARCH_NVIDIA_BLACKWELL,
    GAFIME_GPU_ARCH_NVIDIA_HOPPER, GAFIME_GPU_ARCH_NVIDIA_TURING, GAFIME_GPU_ARCH_UNKNOWN,
    GAFIME_GPU_DEVICE_FLAG_AMD_CDNA, GAFIME_GPU_DEVICE_FLAG_AMD_RDNA,
    GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY, GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION,
    GAFIME_GPU_DEVICE_FLAG_DISCRETE, GAFIME_GPU_DEVICE_FLAG_F64_STORAGE,
    GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH, GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL,
    GAFIME_GPU_DEVICE_FLAG_INTEGRATED, GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY,
    GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64, GAFIME_GPU_DEVICE_FLAG_OPTIX_RT,
    GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuArchitectureClass {
    NvidiaTuring,
    NvidiaAmpere,
    NvidiaAda,
    NvidiaHopper,
    NvidiaBlackwell,
    AmdRdna,
    AmdCdna,
    Apple,
    VendorSpecific(u64),
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuDeviceProfile {
    pub backend_kind: BackendKind,
    pub architecture: GpuArchitectureClass,
    pub flags: u32,
    pub unified_memory: bool,
    pub integrated: bool,
    pub discrete: bool,
    pub managed_memory: bool,
    pub high_bandwidth: bool,
    pub amd_rdna: bool,
    pub amd_cdna: bool,
    pub apple_family: bool,
    pub optix_rt: bool,
    pub immutable_protocol: bool,
    pub descriptor_generation: bool,
    pub mi_accumulation_fp64: bool,
    pub f64_storage: bool,
}

impl GpuDeviceProfile {
    pub fn from_info(info: &GafimeGpuDeviceInfo) -> Self {
        Self {
            backend_kind: info.backend_kind,
            architecture: architecture_class(info),
            flags: info.flags,
            unified_memory: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_UNIFIED_MEMORY),
            integrated: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_INTEGRATED),
            discrete: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_DISCRETE),
            managed_memory: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_MANAGED_MEMORY),
            high_bandwidth: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_HIGH_BANDWIDTH),
            amd_rdna: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_AMD_RDNA),
            amd_cdna: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_AMD_CDNA),
            apple_family: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_APPLE_FAMILY),
            optix_rt: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_OPTIX_RT),
            immutable_protocol: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_IMMUTABLE_PROTOCOL),
            descriptor_generation: has_device_flag(
                info,
                GAFIME_GPU_DEVICE_FLAG_DESCRIPTOR_GENERATION,
            ),
            mi_accumulation_fp64: has_device_flag(
                info,
                GAFIME_GPU_DEVICE_FLAG_MI_ACCUMULATION_FP64,
            ),
            f64_storage: has_device_flag(info, GAFIME_GPU_DEVICE_FLAG_F64_STORAGE),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecisionPathRtPolicy {
    #[default]
    AllowSmFallback,
    RequireRt,
}

impl DecisionPathRtPolicy {
    pub(crate) fn abi_flags(self) -> u32 {
        match self {
            Self::AllowSmFallback => 0,
            Self::RequireRt => GAFIME_DECISION_PATH_FLAG_REQUIRE_RT,
        }
    }
}

pub fn architecture_class(info: &GafimeGpuDeviceInfo) -> GpuArchitectureClass {
    match info.reserved[0] {
        GAFIME_GPU_ARCH_NVIDIA_TURING => GpuArchitectureClass::NvidiaTuring,
        GAFIME_GPU_ARCH_NVIDIA_AMPERE => GpuArchitectureClass::NvidiaAmpere,
        GAFIME_GPU_ARCH_NVIDIA_ADA => GpuArchitectureClass::NvidiaAda,
        GAFIME_GPU_ARCH_NVIDIA_HOPPER => GpuArchitectureClass::NvidiaHopper,
        GAFIME_GPU_ARCH_NVIDIA_BLACKWELL => GpuArchitectureClass::NvidiaBlackwell,
        GAFIME_GPU_ARCH_AMD_RDNA => GpuArchitectureClass::AmdRdna,
        GAFIME_GPU_ARCH_AMD_CDNA => GpuArchitectureClass::AmdCdna,
        GAFIME_GPU_ARCH_APPLE => GpuArchitectureClass::Apple,
        GAFIME_GPU_ARCH_UNKNOWN => GpuArchitectureClass::Unknown,
        value => GpuArchitectureClass::VendorSpecific(value),
    }
}

pub fn has_device_flag(info: &GafimeGpuDeviceInfo, flag: u32) -> bool {
    (info.flags & flag) != 0
}
