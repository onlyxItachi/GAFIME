pub const GAFIME_GPU_DEVICE_FLAG_OPTIX_RT: u32 = 0x100;

pub const GAFIME_DECISION_PATH_SIGN_LE: u32 = 1;
pub const GAFIME_DECISION_PATH_SIGN_GT: u32 = 2;
pub const GAFIME_DECISION_PATH_FLAG_REQUIRE_RT: u32 = 0x1;
/// Conservative path-count ceiling retained by the shared u32 device ABI.
pub const GAFIME_MAX_DECISION_PATH_COUNT: u32 = u32::MAX / 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GafimeDecisionPathTerm {
    pub feature: u32,
    pub sign: u32,
    pub threshold: f32,
    pub reserved32: u32,
    pub reserved: [u64; 2],
}

impl Default for GafimeDecisionPathTerm {
    fn default() -> Self {
        Self {
            feature: 0,
            sign: GAFIME_DECISION_PATH_SIGN_LE,
            threshold: 0.0,
            reserved32: 0,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GafimeDecisionPathBatch {
    pub abi_version: u32,
    pub path_count: u32,
    pub term_count: u32,
    pub flags: u32,
    pub terms: *const GafimeDecisionPathTerm,
    pub path_offsets: *const u32,
    pub membership_host: *mut f32,
    pub reserved: [u64; 8],
}

impl Default for GafimeDecisionPathBatch {
    fn default() -> Self {
        Self {
            abi_version: super::GAFIME_ABI_VERSION,
            path_count: 0,
            term_count: 0,
            flags: 0,
            terms: core::ptr::null(),
            path_offsets: core::ptr::null(),
            membership_host: core::ptr::null_mut(),
            reserved: [0; 8],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GafimeDecisionPathScoreBatch {
    pub abi_version: u32,
    pub path_count: u32,
    pub term_count: u32,
    pub flags: u32,
    pub terms: *const GafimeDecisionPathTerm,
    pub path_offsets: *const u32,
    pub metric_ids: *const u32,
    pub metric_count: u32,
    pub reserved32: u32,
    pub reserved: [u64; 7],
}

impl Default for GafimeDecisionPathScoreBatch {
    fn default() -> Self {
        Self {
            abi_version: super::GAFIME_ABI_VERSION,
            path_count: 0,
            term_count: 0,
            flags: 0,
            terms: core::ptr::null(),
            path_offsets: core::ptr::null(),
            metric_ids: core::ptr::null(),
            metric_count: 0,
            reserved32: 0,
            reserved: [0; 7],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;
    use memoffset::offset_of;

    const LOCAL_ABI_HEADER: &str = include_str!("../../../src/cuda/rt_abi.hpp");

    #[test]
    fn local_cmake_experiment_abi_layouts_stay_in_lockstep() {
        for needle in [
            "#define GAFIME_GPU_DEVICE_FLAG_OPTIX_RT 0x100u",
            "#define GAFIME_DECISION_PATH_SIGN_LE 1u",
            "#define GAFIME_DECISION_PATH_FLAG_REQUIRE_RT 0x1u",
            "#define GAFIME_MAX_DECISION_PATH_COUNT (UINT32_MAX / 4u)",
            "typedef struct GafimeDecisionPathTerm",
            "typedef struct GafimeDecisionPathBatch",
            "typedef struct GafimeDecisionPathScoreBatch",
            "gafime_gpu_decision_path_membership",
            "gafime_gpu_decision_path_score",
        ] {
            assert!(
                LOCAL_ABI_HEADER.contains(needle),
                "missing local ABI marker"
            );
        }

        assert_eq!(GAFIME_GPU_DEVICE_FLAG_OPTIX_RT, 0x100);
        assert_eq!(GAFIME_DECISION_PATH_SIGN_LE, 1);
        assert_eq!(GAFIME_DECISION_PATH_SIGN_GT, 2);
        assert_eq!(GAFIME_DECISION_PATH_FLAG_REQUIRE_RT, 0x1);
        assert_eq!(GAFIME_MAX_DECISION_PATH_COUNT, u32::MAX / 4);

        assert_eq!(size_of::<GafimeDecisionPathTerm>(), 32);
        assert_eq!(offset_of!(GafimeDecisionPathTerm, threshold), 8);
        assert_eq!(offset_of!(GafimeDecisionPathTerm, reserved), 16);

        assert_eq!(size_of::<GafimeDecisionPathBatch>(), 104);
        assert_eq!(offset_of!(GafimeDecisionPathBatch, terms), 16);
        assert_eq!(offset_of!(GafimeDecisionPathBatch, path_offsets), 24);
        assert_eq!(offset_of!(GafimeDecisionPathBatch, membership_host), 32);
        assert_eq!(offset_of!(GafimeDecisionPathBatch, reserved), 40);

        assert_eq!(size_of::<GafimeDecisionPathScoreBatch>(), 104);
        assert_eq!(offset_of!(GafimeDecisionPathScoreBatch, terms), 16);
        assert_eq!(offset_of!(GafimeDecisionPathScoreBatch, path_offsets), 24);
        assert_eq!(offset_of!(GafimeDecisionPathScoreBatch, metric_ids), 32);
        assert_eq!(offset_of!(GafimeDecisionPathScoreBatch, metric_count), 40);
        assert_eq!(offset_of!(GafimeDecisionPathScoreBatch, reserved), 48);
    }
}
