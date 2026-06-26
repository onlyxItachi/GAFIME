#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsaLevel {
    Scalar,
    Sse42,
    Avx2,
    Avx512,
    Neon,
}

pub fn detect_isa() -> IsaLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return IsaLevel::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return IsaLevel::Avx2;
        }
        if std::is_x86_feature_detected!("sse4.2") {
            return IsaLevel::Sse42;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return IsaLevel::Neon;
    }

    IsaLevel::Scalar
}
