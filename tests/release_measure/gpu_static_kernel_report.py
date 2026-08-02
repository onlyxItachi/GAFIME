#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ARITIES = tuple(range(1, 6))
MI_BINS = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96)
REPORT_KINDS = {
    "continuous",
    "spearman",
    "mi",
    "topk_partial",
    "topk_merge",
    "topk_gather",
}

CONTINUOUS_RE = re.compile(r"score_continuous_chunk_kernel_staticILj(\d+)EE")
SPEARMAN_RE = re.compile(r"score_spearman_chunk_kernel_staticILj(\d+)EE")
MI_RE = re.compile(r"score_mutual_info_chunk_kernel_staticILj(\d+)ELj(\d+)EE")
TOPK_PARTIAL_RE = re.compile(r"select_topk_partials_kernel_staticILb([01])EE")
TOPK_MERGE_RE = re.compile(r"merge_topk_partials_kernel_staticILb([01])EE")


class ReportError(RuntimeError):
    pass


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ReportError(f"required tool is not available in PATH: {name}")
    return path


def run_tool(arguments: list[str]) -> str:
    completed = subprocess.run(
        arguments,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReportError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n{detail}"
        )
    return completed.stdout


def classify_kernel(name: str) -> tuple[str, tuple[int, ...]] | None:
    match = MI_RE.search(name)
    if match:
        return "mi", (int(match.group(1)), int(match.group(2)))
    match = CONTINUOUS_RE.search(name)
    if match:
        return "continuous", (int(match.group(1)),)
    match = SPEARMAN_RE.search(name)
    if match:
        return "spearman", (int(match.group(1)),)
    match = TOPK_PARTIAL_RE.search(name)
    if match:
        return "topk_partial", (int(match.group(1)),)
    match = TOPK_MERGE_RE.search(name)
    if match:
        return "topk_merge", (int(match.group(1)),)
    if "copy_selected_metric_rows_kernel" in name:
        return "topk_gather", ()
    if "select_topk_kernel_static" in name:
        return "topk_legacy", ()
    return None


def parse_cuda_resources(text: str) -> dict[tuple[str, str], dict[str, int]]:
    resources: dict[tuple[str, str], dict[str, int]] = {}
    architecture = "unknown"
    function_name: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        arch_match = re.fullmatch(r"arch\s*=\s*(sm_\d+)", line)
        if arch_match:
            architecture = arch_match.group(1)
            continue
        function_match = re.fullmatch(r"Function\s+(.+):", line)
        if function_match:
            function_name = function_match.group(1)
            continue
        if function_name is None or "REG:" not in line:
            continue
        fields = {
            key.lower(): int(value)
            for key, value in re.findall(r"([A-Z]+):([0-9]+)", line)
        }
        if "reg" in fields:
            resources[(architecture, function_name)] = fields
            function_name = None

    return resources


def parse_cuda_sass(text: str) -> dict[tuple[str, str], int]:
    instructions: dict[tuple[str, str], int] = {}
    architecture = "unknown"
    function_name: str | None = None

    for raw_line in text.splitlines():
        arch_match = re.search(r"\bcode for (sm_\d+)\b", raw_line)
        if arch_match:
            architecture = arch_match.group(1)
            continue
        function_match = re.search(r"\bFunction\s*:\s*(\S+)", raw_line)
        if function_match:
            function_name = function_match.group(1)
            instructions.setdefault((architecture, function_name), 0)
            continue
        if function_name is not None and re.match(
            r"^\s*/\*[0-9a-fA-F]{4,}\*/", raw_line
        ):
            key = (architecture, function_name)
            instructions[key] = instructions.get(key, 0) + 1

    return instructions


def cuda_records(library: Path) -> list[dict[str, object]]:
    cuobjdump = require_tool("cuobjdump")
    resources = parse_cuda_resources(
        run_tool([cuobjdump, "--dump-resource-usage", str(library)])
    )
    instructions = parse_cuda_sass(run_tool([cuobjdump, "--dump-sass", str(library)]))
    records: list[dict[str, object]] = []

    for architecture, name in sorted(set(resources) | set(instructions)):
        classification = classify_kernel(name)
        if classification is None:
            continue
        kind, parameters = classification
        record: dict[str, object] = {
            "arch": architecture,
            "kind": kind,
            "name": name,
            "parameters": parameters,
        }
        record.update(resources.get((architecture, name), {}))
        if (architecture, name) in instructions:
            record["sass"] = instructions[(architecture, name)]
        records.append(record)

    if not records:
        raise ReportError(f"no hardened CUDA kernels were found in {library}")
    return records


HIP_METADATA_FIELDS = {
    ".group_segment_fixed_size": "group",
    ".private_segment_fixed_size": "private",
    ".sgpr_count": "sgpr",
    ".sgpr_spill_count": "sgpr_spill",
    ".vgpr_count": "vgpr",
    ".vgpr_spill_count": "vgpr_spill",
    ".wavefront_size": "wavefront",
}


def parse_hip_metadata(text: str) -> dict[str, dict[str, int]]:
    metadata: dict[str, dict[str, int]] = {}
    for block in re.split(r"^  - (?=\.[a-z_]+:)", text, flags=re.MULTILINE)[1:]:
        name_match = re.search(r"^\s+\.name:\s+(\S+)\s*$", block, flags=re.MULTILINE)
        if name_match is None:
            continue
        fields: dict[str, int] = {}
        for source_name, report_name in HIP_METADATA_FIELDS.items():
            value_match = re.search(
                rf"^\s+{re.escape(source_name)}:\s+(\d+)\s*$",
                block,
                flags=re.MULTILINE,
            )
            if value_match:
                fields[report_name] = int(value_match.group(1))
        metadata[name_match.group(1)] = fields
    return metadata


def parse_hip_symbol_sizes(text: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    pattern = re.compile(
        r"^\s*\d+:\s+[0-9a-fA-F]+\s+(\d+)\s+FUNC\s+\S+\s+\S+\s+\S+\s+(\S+)\s*$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            sizes[match.group(2)] = int(match.group(1))
    return sizes


def parse_text_section_size(text: str) -> int | None:
    pattern = re.compile(
        r"^\s*\[\s*\d+\]\s+\.text\s+\S+\s+[0-9a-fA-F]+\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            return int(match.group(1), 16)
    return None


def select_hip_bundle(bundle_list: str, requested_target: str | None) -> str:
    bundles = [line.strip() for line in bundle_list.splitlines() if "amdgcn" in line]
    if requested_target:
        matches = [
            bundle for bundle in bundles if bundle.endswith(f"--{requested_target}")
        ]
        if len(matches) != 1:
            raise ReportError(
                f"HIP target {requested_target!r} was not found uniquely; available bundles: {bundles}"
            )
        return matches[0]
    if len(bundles) != 1:
        raise ReportError(
            f"--hip-target is required for a multi-target HIP fatbin: {bundles}"
        )
    return bundles[0]


def hip_records(
    library: Path, requested_target: str | None
) -> tuple[str, int | None, list[dict[str, object]]]:
    objcopy = require_tool("llvm-objcopy")
    bundler = require_tool("clang-offload-bundler")
    readelf = require_tool("llvm-readelf")

    with tempfile.TemporaryDirectory(prefix="gafime-static-kernel-") as directory:
        temporary = Path(directory)
        fatbin = temporary / "payload.hipfatbin"
        code_object = temporary / "payload.co"
        run_tool([objcopy, "--dump-section", f".hip_fatbin={fatbin}", str(library)])
        bundle_list = run_tool([bundler, "--list", "--type=o", f"--input={fatbin}"])
        bundle = select_hip_bundle(bundle_list, requested_target)
        run_tool(
            [
                bundler,
                "--unbundle",
                "--type=o",
                f"--targets={bundle}",
                f"--input={fatbin}",
                f"--output={code_object}",
            ]
        )
        notes = run_tool([readelf, "--notes", str(code_object)])
        symbols = run_tool([readelf, "--symbols", "--wide", str(code_object)])
        sections = run_tool([readelf, "--sections", "--wide", str(code_object)])

    metadata = parse_hip_metadata(notes)
    symbol_sizes = parse_hip_symbol_sizes(symbols)
    records: list[dict[str, object]] = []
    for name in sorted(set(metadata) | set(symbol_sizes)):
        classification = classify_kernel(name)
        if classification is None:
            continue
        kind, parameters = classification
        record: dict[str, object] = {
            "kind": kind,
            "name": name,
            "parameters": parameters,
        }
        record.update(metadata.get(name, {}))
        if name in symbol_sizes:
            record["code"] = symbol_sizes[name]
        records.append(record)

    if not records:
        raise ReportError(f"no hardened HIP kernels were found in {library}")
    return bundle, parse_text_section_size(sections), records


def range_text(records: list[dict[str, object]], field: str) -> str:
    values = [int(record[field]) for record in records if field in record]
    if not values:
        return "n/a"
    if len(values) != len(records):
        return "incomplete"
    lower = min(values)
    upper = max(values)
    return str(lower) if lower == upper else f"{lower}..{upper}"


def print_group(
    label: str,
    records: list[dict[str, object]],
    fields: tuple[tuple[str, str], ...],
) -> None:
    details = " ".join(
        f"{display}={range_text(records, field)}" for field, display in fields
    )
    print(f"    {label}: kernels={len(records)} {details}")


def matrix_counts(records: list[dict[str, object]]) -> tuple[int, int, int]:
    continuous = {
        tuple(record["parameters"])
        for record in records
        if record["kind"] == "continuous"
    }
    spearman = {
        tuple(record["parameters"])
        for record in records
        if record["kind"] == "spearman"
    }
    mi = {tuple(record["parameters"]) for record in records if record["kind"] == "mi"}
    return len(continuous), len(spearman), len(mi)


def template_matrix_errors(records: list[dict[str, object]], scope: str) -> list[str]:
    expected_arity = {(arity,) for arity in ARITIES}
    expected_mi = {(arity, bins) for arity in ARITIES for bins in MI_BINS}
    errors: list[str] = []
    for kind, expected in (
        ("continuous", expected_arity),
        ("spearman", expected_arity),
        ("mi", expected_mi),
    ):
        actual = {
            tuple(record["parameters"]) for record in records if record["kind"] == kind
        }
        missing = sorted(expected - actual)
        if missing:
            errors.append(f"{scope}: missing {kind} specializations: {missing}")
    return errors


def topk_errors(
    records: list[dict[str, object]],
    scope: str,
    *,
    require_precision_gathers: bool = False,
) -> list[str]:
    names = [str(record["name"]) for record in records]
    partials = {
        tuple(record["parameters"])
        for record in records
        if record["kind"] == "topk_partial"
    }
    merges = {
        tuple(record["parameters"])
        for record in records
        if record["kind"] == "topk_merge"
    }
    gather_names = [
        str(record["name"]) for record in records if record["kind"] == "topk_gather"
    ]
    errors: list[str] = []
    if partials != {(0,), (1,)}:
        errors.append(
            f"{scope}: expected both partial top-k directions, found {sorted(partials)}"
        )
    if merges != {(0,), (1,)}:
        errors.append(
            f"{scope}: expected both top-k merge directions, found {sorted(merges)}"
        )
    if require_precision_gathers:
        expected_gathers = {
            "legacy-f32": "copy_selected_metric_rows_kernelEPKfPKjmjPf",
            "abi-1.1-f32": ("precision_kernel32copy_selected_metric_rows_kernelIfEE"),
            "abi-1.1-f64": ("precision_kernel32copy_selected_metric_rows_kernelIdEE"),
        }
        for identity, marker in expected_gathers.items():
            count = sum(marker in name for name in gather_names)
            if count != 1:
                errors.append(
                    f"{scope}: expected one {identity} selected-row gather kernel, "
                    f"found {count}"
                )
        if len(gather_names) != len(expected_gathers):
            errors.append(
                f"{scope}: expected exactly {len(expected_gathers)} legacy/typed "
                f"selected-row gather kernels, found {len(gather_names)}"
            )
    elif len(gather_names) != 1:
        errors.append(
            f"{scope}: expected one selected-row gather kernel, found "
            f"{len(gather_names)}"
        )
    if any("select_topk_kernel_static" in name for name in names):
        errors.append(
            f"{scope}: legacy single-block select_topk_kernel_static is still present"
        )
    return errors


def metadata_errors(
    records: list[dict[str, object]],
    fields: tuple[str, ...],
    scope: str,
) -> list[str]:
    errors: list[str] = []
    for record in records:
        if record["kind"] not in REPORT_KINDS:
            continue
        missing = [field for field in fields if field not in record]
        if missing:
            errors.append(
                f"{scope}: {record['name']} is missing metadata fields {missing}"
            )
    return errors


def no_spill_errors(
    records: list[dict[str, object]],
    fields: tuple[str, ...],
    scope: str,
) -> list[str]:
    errors: list[str] = []
    for field in fields:
        count = sum(
            record["kind"] in REPORT_KINDS and int(record.get(field, 0)) != 0
            for record in records
        )
        if count:
            errors.append(f"{scope}: {count} hardened kernels have nonzero {field}")
    return errors


def print_backend_report(
    label: str,
    records: list[dict[str, object]],
    fields: tuple[tuple[str, str], ...],
) -> None:
    continuous_count, spearman_count, mi_count = matrix_counts(records)
    print(
        "    template matrix: "
        f"continuous={continuous_count}/{len(ARITIES)} "
        f"spearman={spearman_count}/{len(ARITIES)} "
        f"mi={mi_count}/{len(ARITIES) * len(MI_BINS)}"
    )
    for kind, title in (
        ("continuous", "continuous arity=1..5"),
        ("spearman", "spearman arity=1..5"),
    ):
        selected = [record for record in records if record["kind"] == kind]
        print_group(title, selected, fields)
    for bins in MI_BINS:
        selected = [
            record
            for record in records
            if record["kind"] == "mi" and tuple(record["parameters"])[1] == bins
        ]
        print_group(f"mi bins={bins}", selected, fields)
    for kind, title in (
        ("topk_partial", "top-k partials"),
        ("topk_merge", "top-k merge"),
        ("topk_gather", "selected-row gather"),
    ):
        selected = [record for record in records if record["kind"] == kind]
        print_group(title, selected, fields)
    print(f"    source: {label}")


def existing_library(value: str | None, option: str) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ReportError(f"{option} does not name a file: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report CUDA SASS and HIP code-object kernel pressure without running a GPU."
    )
    parser.add_argument("--cuda-lib", help="CUDA shared library containing .nv_fatbin")
    parser.add_argument("--hip-lib", help="HIP shared library containing .hip_fatbin")
    parser.add_argument(
        "--hip-target", help="HIP bundle architecture, for example gfx1150"
    )
    parser.add_argument(
        "--require-template-matrix",
        action="store_true",
        help="fail unless every arity and MI-bin specialization is emitted",
    )
    parser.add_argument(
        "--require-topk-split",
        action="store_true",
        help="fail unless two-stage ascending/descending top-k and selected-row gather are emitted",
    )
    parser.add_argument(
        "--require-no-spills",
        action="store_true",
        help="fail on CUDA local/stack use or HIP private/VGPR/SGPR spills",
    )
    arguments = parser.parse_args()

    try:
        cuda_library = existing_library(arguments.cuda_lib, "--cuda-lib")
        hip_library = existing_library(arguments.hip_lib, "--hip-lib")
        if cuda_library is None and hip_library is None:
            parser.error("at least one of --cuda-lib or --hip-lib is required")

        failures: list[str] = []
        if cuda_library is not None:
            records = cuda_records(cuda_library)
            print("CUDA static kernel report")
            print(f"  artifact: {cuda_library} ({cuda_library.stat().st_size:,} bytes)")
            for architecture in sorted({str(record["arch"]) for record in records}):
                selected = [
                    record for record in records if record["arch"] == architecture
                ]
                print(f"  architecture: {architecture}")
                print_backend_report(
                    str(cuda_library),
                    selected,
                    (
                        ("sass", "sass"),
                        ("reg", "reg"),
                        ("shared", "shared"),
                        ("stack", "stack"),
                        ("local", "local"),
                    ),
                )
                failures.extend(
                    metadata_errors(
                        selected,
                        ("sass", "reg", "shared", "stack", "local"),
                        f"CUDA {architecture}",
                    )
                )
                if arguments.require_template_matrix:
                    failures.extend(
                        template_matrix_errors(selected, f"CUDA {architecture}")
                    )
                if arguments.require_topk_split:
                    failures.extend(topk_errors(selected, f"CUDA {architecture}"))
                if arguments.require_no_spills:
                    failures.extend(
                        no_spill_errors(
                            selected,
                            ("stack", "local"),
                            f"CUDA {architecture}",
                        )
                    )

        if hip_library is not None:
            bundle, text_size, records = hip_records(hip_library, arguments.hip_target)
            if cuda_library is not None:
                print()
            print("HIP static kernel report")
            print(f"  artifact: {hip_library} ({hip_library.stat().st_size:,} bytes)")
            print(f"  bundle: {bundle}")
            if text_size is not None:
                print(f"  code-object .text: {text_size:,} bytes (0x{text_size:x})")
            print_backend_report(
                str(hip_library),
                records,
                (
                    ("code", "code"),
                    ("vgpr", "vgpr"),
                    ("sgpr", "sgpr"),
                    ("group", "group"),
                    ("private", "private"),
                    ("vgpr_spill", "vgpr_spill"),
                    ("sgpr_spill", "sgpr_spill"),
                    ("wavefront", "wave"),
                ),
            )
            failures.extend(
                metadata_errors(
                    records,
                    (
                        "code",
                        "vgpr",
                        "sgpr",
                        "group",
                        "private",
                        "vgpr_spill",
                        "sgpr_spill",
                        "wavefront",
                    ),
                    f"HIP {bundle}",
                )
            )
            if arguments.require_template_matrix:
                failures.extend(template_matrix_errors(records, f"HIP {bundle}"))
            if arguments.require_topk_split:
                failures.extend(
                    topk_errors(
                        records,
                        f"HIP {bundle}",
                        require_precision_gathers=True,
                    )
                )
            if arguments.require_no_spills:
                failures.extend(
                    no_spill_errors(
                        records,
                        ("private", "vgpr_spill", "sgpr_spill"),
                        f"HIP {bundle}",
                    )
                )

        if failures:
            print("\nStatic artifact checks: FAIL", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1
        print("\nStatic artifact checks: PASS")
        return 0
    except ReportError as error:
        print(f"gpu_static_kernel_report.py: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
