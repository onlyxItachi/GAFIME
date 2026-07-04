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

pub fn finite_dispatch_isa() -> IsaLevel {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_dispatch_reports_detected_simd_when_available() {
        let isa = finite_dispatch_isa();
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                assert_eq!(isa, IsaLevel::Avx512);
            } else if std::is_x86_feature_detected!("avx2") {
                assert_eq!(isa, IsaLevel::Avx2);
            } else if std::is_x86_feature_detected!("sse4.2") {
                assert_eq!(isa, IsaLevel::Sse42);
            }
        }
        #[cfg(target_arch = "aarch64")]
        assert_eq!(isa, IsaLevel::Neon);
    }
}
