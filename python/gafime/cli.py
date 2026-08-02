from __future__ import annotations

import argparse

from gafime import __version__, backend_capabilities


def main() -> int:
    parser = argparse.ArgumentParser(description="GAFIME v1 native runtime")
    parser.add_argument(
        "-V", "--version", action="version", version=f"gafime {__version__}"
    )
    parser.add_argument(
        "--check", action="store_true", help="Report native/backend capabilities"
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="Backend to inspect: auto, core, cuda, rocm, hip, or metal (default: auto)",
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="Device index to inspect"
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "mixed", "fp64"),
        default="mixed",
        help="Precision profile to inspect (default: mixed)",
    )
    args = parser.parse_args()

    if args.check:
        return _check_v1_boundary(args.backend, args.device_id, args.precision)
    parser.print_help()
    return 0


def _check_v1_boundary(backend: str, device_id: int, precision: str = "mixed") -> int:
    capabilities = backend_capabilities(
        backend, device_id, probe=True, precision=precision
    )
    print(f"GAFIME package: {__version__}")
    print(f"native boundary: {_display(capabilities.native_boundary.value)}")
    print(f"native version: {_display(capabilities.native_version.value)}")
    print(f"configured backend: {capabilities.configured_backend}")
    print(f"selected backend: {_display(capabilities.selected_backend)}")
    print(f"backend status: {capabilities.selection_status}")
    print(
        f"runtime probe: {'performed' if capabilities.probe_performed else 'not performed'}"
    )
    if capabilities.selection_detail:
        print(f"resolution: {capabilities.selection_detail}")

    for name, detail in sorted(capabilities.probe_details.items()):
        if not isinstance(detail, dict):
            continue
        print(
            f"candidate {name}: {detail.get('status', 'unknown')}"
            + (f" ({detail['detail']})" if detail.get("detail") else "")
        )

    graph = capabilities.graph_support.value
    if isinstance(graph, dict):
        print(
            f"graph support: {_display(graph.get('supported'))} ({graph.get('mode')})"
        )
    else:
        print(f"graph support: {_display(graph)}")
    print(f"device significance: {_display(capabilities.device_significance.value)}")
    print(
        f"host significance fallback: {_display(capabilities.host_significance_fallback.value)}"
    )
    print(
        f"permutation significance: {_display(capabilities.permutation_significance.value)}"
    )
    print(
        f"stability significance: {_display(capabilities.stability_significance.value)}"
    )
    mi = capabilities.mi_bin_ceiling.value
    if isinstance(mi, dict):
        print(
            "MI: "
            f"{_display(capabilities.mi_estimator.value)}, "
            f"template ceiling={mi['effective_template_ceiling']} "
            f"(backend max={mi['backend_max']})"
        )
    else:
        print(f"MI: {_display(capabilities.mi_estimator.value)}")
    precision = capabilities.precision_contract.value
    if isinstance(precision, dict):
        print(
            "precision: "
            f"requested={precision['requested']}; "
            f"effective={_display(precision['effective'])}; "
            f"accumulators={precision['accumulators']}"
        )
    payload_policy = capabilities.payload_build_policy.value
    if isinstance(payload_policy, dict):
        policy_name = payload_policy.get("wheel_policy", "package-declared")
        print(
            "payload build policy: "
            f"{policy_name} ({capabilities.payload_build_policy.source})"
        )
        if payload_policy.get("mixed_runtime_coexistence"):
            print(
                "payload runtime coexistence: "
                f"{payload_policy['mixed_runtime_coexistence']}"
            )
    else:
        print("payload build policy: unknown")
    arrow = capabilities.arrow_ingest_mode.value
    if isinstance(arrow, dict):
        print(
            "Arrow ingest: "
            f"{arrow['protocol']}; {arrow['record_batches']}; {arrow['compute_buffer']}"
        )
    for family in capabilities.families:
        significance = family.significance_support
        print(
            f"family {family.name}: generation={family.generation_placement}; "
            f"scoring={','.join(family.scoring_backends)}; graph={family.graph_scope}; "
            f"significance=permutation:{significance.permutation},"
            f"stability:{significance.stability}"
        )

    return 0 if capabilities.selection_status == "available" else 1


def _display(value: object) -> str:
    return "unknown" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
