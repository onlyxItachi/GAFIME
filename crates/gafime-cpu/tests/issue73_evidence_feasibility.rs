//! Run the branch-only experiment's regression fixtures in normal workspace CI.
//! Neither module is part of the shipping Core library or Python public API.

#[path = "../examples/issue73_probe/probe.rs"]
mod probe;
#[path = "../examples/issue73_probe/tests.rs"]
mod tests;
