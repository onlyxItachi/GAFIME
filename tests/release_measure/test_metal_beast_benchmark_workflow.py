"""Static trust-boundary checks for the Metal benchmark workflow.

These tests inspect workflow structure only; they never dispatch or execute the
hosted benchmark.
"""

from pathlib import Path


WORKFLOW = (
    Path(__file__).parents[2] / ".github" / "workflows" / "metal_beast_benchmark.yml"
)


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_metal_candidate_is_authorized_before_checkout_in_a_trusted_context() -> None:
    workflow = _workflow()
    setup = workflow.index("- name: Set up Python")
    paths = workflow.index("- name: Prepare external benchmark paths")
    preflight = workflow.index(
        "- name: Require exact live PR head before candidate checkout"
    )
    checkout = workflow.index("- uses: actions/checkout@")
    binding = workflow.index("- name: Bind exact candidate checkout")

    assert setup < paths < preflight < checkout < binding

    preflight_step = workflow[preflight:checkout]
    assert "working-directory: ${{ runner.temp }}" in preflight_step
    assert "python -I - <<'PY'" in preflight_step
    assert "import subprocess" not in preflight_step
    for predicate in (
        'live_head.get("repo", {}).get("full_name") != repository',
        'live_head.get("ref") != "codex/issue-25-three-precision-profiles"',
        'if pull.get("state") != "open"',
        "if live_sha != expected",
    ):
        assert predicate in preflight_step
    assert 'record["status"]' not in preflight_step

    bind_step = workflow[binding : workflow.index("- name: Collect host specs JSON")]
    assert "working-directory: ${{ runner.temp }}" in bind_step
    assert "python -I - <<'PY'" in bind_step
    assert '["git", "-C", workspace, "rev-parse", "HEAD"]' in bind_step
    assert (
        '["git", "-C", workspace, "status", "--porcelain=v1", "--untracked-files=all"]'
        in bind_step
    )
    assert 'record["checkout_status"] = "pass"' in bind_step
    assert 'record["status"] = "pass"' in bind_step


def test_metal_failure_artifacts_stay_under_fixed_runner_temp_paths() -> None:
    workflow = _workflow()
    prepare = workflow[
        workflow.index("- name: Prepare external benchmark paths") : workflow.index(
            "- name: Require exact live PR head before candidate checkout"
        )
    ]
    assert "id: prepare_external_benchmark_paths" in prepare
    assert 'mkdir -p "$RUNNER_TEMP/gafime-metal-beast-results"' in prepare

    summary = workflow[
        workflow.index("- name: Summarize benchmark") : workflow.index(
            "- name: Upload Metal beast artifacts"
        )
    ]
    assert "steps.bind_exact_candidate_checkout.outcome == 'success'" in summary
    assert "working-directory: ${{ runner.temp }}" in summary
    assert "python -I - <<'PY'" in summary

    upload = workflow[workflow.index("- name: Upload Metal beast artifacts") :]
    assert "steps.prepare_external_benchmark_paths.outcome == 'success'" in upload
    assert "${{ env.METAL_RESULTS_DIR }}" not in upload
    assert "${{ env.METAL_WHEELHOUSE }}" not in upload
    assert "${{ env.METAL_BASELINE_WHEELHOUSE }}" not in upload
    for path in (
        "${{ runner.temp }}/gafime-metal-beast-results/**",
        "${{ runner.temp }}/gafime-metal-candidate-wheelhouse/*.whl",
        "${{ runner.temp }}/gafime-metal-baseline-wheelhouse/*.whl",
    ):
        assert path in upload
    assert "if-no-files-found: warn" in upload
