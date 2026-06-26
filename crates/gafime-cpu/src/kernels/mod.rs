#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricKernel {
    Pearson,
    Spearman,
    MutualInfo,
    R2,
}

pub fn planned_kernel_names() -> &'static [&'static str] {
    &["pearson", "spearman", "mutual_info", "r2"]
}
