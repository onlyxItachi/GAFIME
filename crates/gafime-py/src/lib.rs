pub const BOUNDARY_NAME: &str = "gafime-py";

pub fn boundary_name() -> &'static str {
    BOUNDARY_NAME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_name_is_stable() {
        assert_eq!(boundary_name(), "gafime-py");
    }
}
