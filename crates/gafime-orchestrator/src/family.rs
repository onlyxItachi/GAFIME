use gafime_types::{
    CandidateFamily, GAFIME_FAMILY_CONTINUOUS, GAFIME_FAMILY_DECISION_PATH,
    GAFIME_FAMILY_TIME_SERIES,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FamilyDescriptor {
    pub family: CandidateFamily,
    pub name: &'static str,
    pub continuous_input: bool,
    pub cpu_kernel: bool,
    pub cuda_kernel: bool,
    pub rocm_kernel: bool,
    pub python_candidate_loop: bool,
}

impl FamilyDescriptor {
    pub fn supported_on_any_device(&self) -> bool {
        self.cpu_kernel || self.cuda_kernel || self.rocm_kernel
    }
}

pub const FAMILY_DESCRIPTORS: &[FamilyDescriptor] = &[
    FamilyDescriptor {
        family: GAFIME_FAMILY_CONTINUOUS,
        name: "continuous",
        continuous_input: true,
        cpu_kernel: true,
        cuda_kernel: true,
        rocm_kernel: false,
        python_candidate_loop: false,
    },
    FamilyDescriptor {
        family: GAFIME_FAMILY_DECISION_PATH,
        name: "decision_path",
        continuous_input: true,
        // Implemented by native GBDT split-finding (depth-k conjunction paths +
        // residual boosting) that materializes membership columns, then continuous
        // mining, so it runs on whichever backend scores the continuous chunks.
        cpu_kernel: true,
        cuda_kernel: true,
        rocm_kernel: false,
        python_candidate_loop: false,
    },
    FamilyDescriptor {
        family: GAFIME_FAMILY_TIME_SERIES,
        name: "time_series",
        continuous_input: true,
        // Implemented by feature-expansion (lag/window/velocity) + continuous
        // mining, so it runs on whichever backend scores the continuous chunks.
        cpu_kernel: true,
        cuda_kernel: true,
        rocm_kernel: false,
        python_candidate_loop: false,
    },
];

pub fn family_descriptors() -> &'static [FamilyDescriptor] {
    FAMILY_DESCRIPTORS
}

pub fn descriptor_for(family: CandidateFamily) -> Option<&'static FamilyDescriptor> {
    FAMILY_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.family == family)
}

pub fn descriptor_by_name(name: &str) -> Option<&'static FamilyDescriptor> {
    FAMILY_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_families_have_rust_contracts_without_python_loops() {
        let names = family_descriptors()
            .iter()
            .map(|descriptor| descriptor.name)
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["continuous", "decision_path", "time_series"]);
        assert!(family_descriptors()
            .iter()
            .all(|descriptor| !descriptor.python_candidate_loop));
    }

    #[test]
    fn family_support_reflects_implemented_kernels() {
        let decision_path = descriptor_by_name("decision_path").unwrap();
        let time_series = descriptor_by_name("time_series").unwrap();

        // Both non-continuous families are implemented by feature-expansion +
        // continuous mining, so both run on CPU and CUDA.
        assert!(time_series.supported_on_any_device());
        assert!(time_series.cpu_kernel && time_series.cuda_kernel);
        assert!(decision_path.supported_on_any_device());
        assert!(decision_path.cpu_kernel && decision_path.cuda_kernel);
        assert_eq!(
            descriptor_for(GAFIME_FAMILY_DECISION_PATH),
            Some(decision_path)
        );
        assert_eq!(descriptor_for(GAFIME_FAMILY_TIME_SERIES), Some(time_series));
    }
}
