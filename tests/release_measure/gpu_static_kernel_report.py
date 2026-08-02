#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
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

PRECISION_SPECIALIZATION_KINDS = ("continuous", "spearman", "mi")
PRECISION_TYPE_ENCODINGS = {
    "fp32": "fff",
    "mixed": "fdd",
    "fp64": "ddd",
}
PRECISION_KERNEL_RE = re.compile(
    r"precision_kernel(?:"
    r"17continuous_kernel|"
    r"15spearman_kernel|"
    r"18mutual_info_kernel|"
    r"36score_continuous_chunk_kernel_static|"
    r"34score_spearman_chunk_kernel_static|"
    r"37score_mutual_info_chunk_kernel_static"
    r")I(fff|fdd|ddd)"
)
PRECISION_EVIDENCE_SCHEMA = 1

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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def precision_specialization(name: str) -> tuple[str, str] | None:
    """Classify a typed device-kernel symbol by arithmetic profile and metric."""

    match = PRECISION_KERNEL_RE.search(name)
    if match is None:
        return None
    encoding = match.group(1)
    profile = next(
        profile
        for profile, candidate in PRECISION_TYPE_ENCODINGS.items()
        if candidate == encoding
    )
    if "mutual_info" in name:
        kind = "mi"
    elif "spearman" in name:
        kind = "spearman"
    else:
        kind = "continuous"
    return profile, kind


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
        specialization = precision_specialization(name)
        if classification is None and specialization is None:
            continue
        if classification is None:
            assert specialization is not None
            profile, precision_kind = specialization
            kind, parameters = f"precision_{precision_kind}", ()
        else:
            kind, parameters = classification
        record: dict[str, object] = {
            "arch": architecture,
            "kind": kind,
            "name": name,
            "parameters": parameters,
        }
        if specialization is not None:
            record["profile"] = specialization[0]
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
        inspection_copy = temporary / "payload-inspection.so"
        source_digest = file_sha256(library)
        run_tool(
            [
                objcopy,
                "--dump-section",
                f".hip_fatbin={fatbin}",
                str(library),
                str(inspection_copy),
            ]
        )
        if file_sha256(library) != source_digest:
            raise ReportError(
                f"HIP static inspection mutated its input artifact: {library}"
            )
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
        specialization = precision_specialization(name)
        if classification is None and specialization is None:
            continue
        if classification is None:
            assert specialization is not None
            profile, precision_kind = specialization
            kind, parameters = f"precision_{precision_kind}", ()
        else:
            kind, parameters = classification
        record: dict[str, object] = {
            "kind": kind,
            "name": name,
            "parameters": parameters,
        }
        if specialization is not None:
            record["profile"] = specialization[0]
        record.update(metadata.get(name, {}))
        if name in symbol_sizes:
            record["code"] = symbol_sizes[name]
        records.append(record)

    if not records:
        raise ReportError(f"no hardened HIP kernels were found in {library}")
    return bundle, parse_text_section_size(sections), records


def hip_targets(library: Path) -> tuple[str, ...]:
    objcopy = require_tool("llvm-objcopy")
    bundler = require_tool("clang-offload-bundler")
    with tempfile.TemporaryDirectory(prefix="gafime-static-kernel-list-") as directory:
        temporary = Path(directory)
        fatbin = temporary / "payload.hipfatbin"
        inspection_copy = temporary / "payload-inspection.so"
        source_digest = file_sha256(library)
        run_tool(
            [
                objcopy,
                "--dump-section",
                f".hip_fatbin={fatbin}",
                str(library),
                str(inspection_copy),
            ]
        )
        if file_sha256(library) != source_digest:
            raise ReportError(
                f"HIP static inspection mutated its input artifact: {library}"
            )
        bundle_list = run_tool([bundler, "--list", "--type=o", f"--input={fatbin}"])
    bundles = [line.strip() for line in bundle_list.splitlines() if "amdgcn" in line]
    targets = tuple(sorted(bundle.rsplit("--", 1)[-1] for bundle in bundles))
    if not targets or len(targets) != len(set(targets)):
        raise ReportError(f"HIP bundle target list is invalid: {bundles}")
    return targets


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


def precision_specialization_summary(
    records: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    summary = {
        profile: {kind: 0 for kind in PRECISION_SPECIALIZATION_KINDS}
        for profile in PRECISION_TYPE_ENCODINGS
    }
    for record in records:
        profile = record.get("profile")
        kind = str(record.get("kind", "")).removeprefix("precision_")
        machine_code = record.get("sass", record.get("code", 0))
        if (
            profile in summary
            and kind in summary[str(profile)]
            and isinstance(machine_code, int)
            and machine_code > 0
        ):
            summary[str(profile)][kind] += 1
    return summary


def precision_specialization_errors(
    records: list[dict[str, object]], scope: str
) -> list[str]:
    summary = precision_specialization_summary(records)
    errors: list[str] = []
    for profile, kinds in summary.items():
        missing = [kind for kind, count in kinds.items() if count == 0]
        if missing:
            errors.append(
                f"{scope}: {profile} is missing typed device machine code for {missing}"
            )
    return errors


def print_precision_specializations(records: list[dict[str, object]]) -> None:
    summary = precision_specialization_summary(records)
    detail = " ".join(
        f"{profile}="
        + ",".join(f"{kind}:{count}" for kind, count in kinds.items())
        for profile, kinds in summary.items()
    )
    print(f"    precision device specializations: {detail}")


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
    print_precision_specializations(records)
    print(f"    source: {label}")


def existing_library(value: str | None, option: str) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ReportError(f"{option} does not name a file: {path}")
    return path


def _target_profile_evidence(
    target: str, records: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "target": target,
        "profiles": precision_specialization_summary(records),
    }


def _write_precision_evidence(
    directory: Path,
    backend: str,
    library: Path,
    wheel: Path,
    targets: list[dict[str, object]],
) -> Path:
    native_sha256 = file_sha256(library)
    native_member, wheel_native = _wheel_native_member(wheel, backend)
    wheel_native_sha256 = hashlib.sha256(wheel_native).hexdigest()
    if wheel_native_sha256 != native_sha256 or len(wheel_native) != library.stat().st_size:
        raise ReportError(
            f"{wheel.name} extracted {native_member} does not match inspected "
            f"library {library}"
        )
    wheel_sha256 = file_sha256(wheel)
    record = {
        "schema_version": PRECISION_EVIDENCE_SCHEMA,
        "backend": backend,
        "native_sha256": native_sha256,
        "native_size": library.stat().st_size,
        "wheel": {
            "filename": wheel.name,
            "native_member": native_member,
            "sha256": wheel_sha256,
        },
        "targets": sorted(targets, key=lambda item: str(item["target"])),
    }
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{backend}-{wheel_sha256}.precision-evidence.json"
    contents = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != contents:
        raise ReportError(f"precision evidence hash collision at {output}")
    output.write_text(contents, encoding="utf-8")
    print(f"  wheel-sha256: {wheel_sha256}")
    print(f"  native-member: {native_member}")
    print(f"  native-sha256: {native_sha256}")
    print(f"  evidence: {output}")
    return output


def _wheel_native_member(wheel: Path, backend: str) -> tuple[str, bytes]:
    expected = {
        "cuda": {
            "gafime_cuda/libgafime_cuda.so",
            "gafime_cuda/gafime_cuda.dll",
            "gafime_cuda/gafime_cuda.pyd",
        },
        "rocm": {"gafime_rocm/libgafime_rocm.so"},
    }[backend]
    with zipfile.ZipFile(wheel) as archive:
        matches = sorted(set(archive.namelist()) & expected)
        if len(matches) != 1:
            raise ReportError(
                f"{wheel.name} expected one {backend} native member, found {matches}"
            )
        return matches[0], archive.read(matches[0])


def _expected_wheel_targets(wheel: Path, backend: str) -> set[str]:
    if backend == "rocm":
        policy = json.loads(
            (ROOT / ".github" / "scripts" / "rocm_7_2_3_system_policy.json").read_text(
                encoding="utf-8"
            )
        )
        return set(policy["gfx_targets"])
    with zipfile.ZipFile(wheel) as archive:
        policy_names = [
            name for name in archive.namelist() if name == "gafime_cuda/build_policy.json"
        ]
        if len(policy_names) != 1:
            raise ReportError(f"{wheel.name} has no unique CUDA build policy")
        policy = json.loads(archive.read(policy_names[0]).decode("utf-8"))
    return {f"sm_{architecture}" for architecture in policy["cuda_architectures"]}


def _validate_evidence_record(record: object, path: Path) -> tuple[str, str]:
    if not isinstance(record, dict):
        raise ReportError(f"{path} precision evidence must be an object")
    backend = record.get("backend")
    native_sha256 = record.get("native_sha256")
    if (
        record.get("schema_version") != PRECISION_EVIDENCE_SCHEMA
        or backend not in {"cuda", "rocm"}
        or not isinstance(native_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", native_sha256) is None
        or not isinstance(record.get("native_size"), int)
        or int(record["native_size"]) <= 0
        or not isinstance(record.get("targets"), list)
        or not isinstance(record.get("wheel"), dict)
    ):
        raise ReportError(f"{path} precision evidence identity is invalid")
    wheel = record["wheel"]
    if (
        set(wheel) != {"filename", "native_member", "sha256"}
        or not isinstance(wheel["filename"], str)
        or not isinstance(wheel["native_member"], str)
        or not isinstance(wheel["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", wheel["sha256"]) is None
    ):
        raise ReportError(f"{path} wheel binding is invalid: {wheel!r}")
    for target in record["targets"]:
        if not isinstance(target, dict) or set(target) != {"target", "profiles"}:
            raise ReportError(f"{path} has an invalid target record: {target!r}")
        profiles = target["profiles"]
        if not isinstance(profiles, dict) or set(profiles) != set(
            PRECISION_TYPE_ENCODINGS
        ):
            raise ReportError(f"{path} has an invalid profile map: {profiles!r}")
        for profile, kinds in profiles.items():
            if (
                not isinstance(kinds, dict)
                or set(kinds) != set(PRECISION_SPECIALIZATION_KINDS)
                or any(not isinstance(count, int) or count <= 0 for count in kinds.values())
            ):
                raise ReportError(
                    f"{path} {target['target']}/{profile} lacks typed machine code: "
                    f"{kinds!r}"
                )
    return str(backend), str(wheel["sha256"])


def verify_wheel_evidence(wheels: Path, evidence: Path) -> None:
    evidence_paths = sorted(evidence.rglob("*.precision-evidence.json"))
    if not evidence_paths:
        raise ReportError(f"no precision evidence found under {evidence}")
    records: dict[tuple[str, str], tuple[Path, dict[str, object]]] = {}
    for path in evidence_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        backend, wheel_sha256 = _validate_evidence_record(record, path)
        key = (backend, wheel_sha256)
        if key in records:
            raise ReportError(
                f"duplicate precision evidence for {backend}/{wheel_sha256}: "
                f"{records[key][0]}, {path}"
            )
        records[key] = (path, record)

    wheel_paths = sorted(
        path
        for path in wheels.rglob("*.whl")
        if path.name.startswith(("gafime_cuda-", "gafime_rocm-"))
    )
    if not wheel_paths:
        raise ReportError(f"no CUDA or ROCm wheels found under {wheels}")
    used: set[Path] = set()
    for wheel in wheel_paths:
        backend = "cuda" if wheel.name.startswith("gafime_cuda-") else "rocm"
        native_member, native = _wheel_native_member(wheel, backend)
        native_sha256 = hashlib.sha256(native).hexdigest()
        wheel_sha256 = file_sha256(wheel)
        key = (backend, wheel_sha256)
        if key not in records:
            raise ReportError(
                f"{wheel.name} has no exact wheel-hash machine-code evidence "
                f"({wheel_sha256})"
            )
        path, record = records[key]
        used.add(path)
        wheel_record = record["wheel"]
        if (
            wheel_record["filename"] != wheel.name
            or wheel_record["native_member"] != native_member
            or record["native_sha256"] != native_sha256
            or record["native_size"] != len(native)
        ):
            raise ReportError(
                f"{wheel.name} extracted native member differs from {path.name} evidence"
            )
        actual_targets = {str(item["target"]) for item in record["targets"]}
        expected_targets = _expected_wheel_targets(wheel, backend)
        if actual_targets != expected_targets:
            raise ReportError(
                f"{wheel.name} machine-code targets {sorted(actual_targets)} != "
                f"{sorted(expected_targets)}"
            )
        print(
            "WHEEL PRECISION EVIDENCE: "
            f"{wheel.name} wheel_sha256={wheel_sha256} "
            f"native_member={native_member} native_sha256={native_sha256} "
            f"targets={len(actual_targets)} profiles=fp32,mixed,fp64"
        )
    unused = set(evidence_paths) - used
    if unused:
        raise ReportError(f"unused or stale precision evidence: {sorted(map(str, unused))}")
    print(
        "WHEEL PRECISION EVIDENCE: PASS "
        f"wheels={len(wheel_paths)} evidence_records={len(evidence_paths)}"
    )


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
        "--hip-all-targets",
        action="store_true",
        help="inspect every HIP code-object target carried by the library",
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
    parser.add_argument(
        "--require-precision-profiles",
        action="store_true",
        help="fail unless fp32, mixed, and fp64 typed metric device code is emitted",
    )
    parser.add_argument(
        "--write-evidence-dir",
        type=Path,
        help="write hash-bound machine-code evidence after all requested checks pass",
    )
    parser.add_argument(
        "--evidence-wheel",
        type=Path,
        help="exact repaired wheel containing the inspected native library",
    )
    parser.add_argument(
        "--verify-wheel-evidence",
        type=Path,
        help="verify exact CUDA/ROCm wheels against previously generated evidence",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="evidence directory used with --verify-wheel-evidence",
    )
    arguments = parser.parse_args()

    try:
        if arguments.verify_wheel_evidence is not None:
            if arguments.evidence_dir is None:
                parser.error("--verify-wheel-evidence requires --evidence-dir")
            if (
                arguments.cuda_lib is not None
                or arguments.hip_lib is not None
                or arguments.evidence_wheel is not None
                or arguments.write_evidence_dir is not None
            ):
                parser.error("wheel-evidence verification cannot inspect a library")
            verify_wheel_evidence(
                arguments.verify_wheel_evidence.resolve(),
                arguments.evidence_dir.resolve(),
            )
            return 0
        if arguments.evidence_dir is not None:
            parser.error("--evidence-dir is only valid with --verify-wheel-evidence")
        if arguments.hip_all_targets and arguments.hip_target is not None:
            parser.error("--hip-all-targets and --hip-target are mutually exclusive")
        if (
            arguments.write_evidence_dir is not None
            and not arguments.require_precision_profiles
        ):
            parser.error(
                "--write-evidence-dir requires --require-precision-profiles"
            )
        if (arguments.write_evidence_dir is None) != (arguments.evidence_wheel is None):
            parser.error(
                "--write-evidence-dir and --evidence-wheel must be supplied together"
            )
        evidence_wheel = (
            existing_library(str(arguments.evidence_wheel), "--evidence-wheel")
            if arguments.evidence_wheel is not None
            else None
        )
        cuda_library = existing_library(arguments.cuda_lib, "--cuda-lib")
        hip_library = existing_library(arguments.hip_lib, "--hip-lib")
        if cuda_library is None and hip_library is None:
            parser.error("at least one of --cuda-lib or --hip-lib is required")

        failures: list[str] = []
        cuda_evidence: list[dict[str, object]] = []
        hip_evidence: list[dict[str, object]] = []
        if cuda_library is not None:
            records = cuda_records(cuda_library)
            print("CUDA static kernel report")
            print(f"  artifact: {cuda_library} ({cuda_library.stat().st_size:,} bytes)")
            for architecture in sorted({str(record["arch"]) for record in records}):
                selected = [
                    record for record in records if record["arch"] == architecture
                ]
                cuda_evidence.append(_target_profile_evidence(architecture, selected))
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
                if arguments.require_precision_profiles:
                    failures.extend(
                        precision_specialization_errors(
                            selected, f"CUDA {architecture}"
                        )
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
            if cuda_library is not None:
                print()
            print("HIP static kernel report")
            print(f"  artifact: {hip_library} ({hip_library.stat().st_size:,} bytes)")
            targets: tuple[str | None, ...]
            if arguments.hip_all_targets:
                targets = hip_targets(hip_library)
            else:
                targets = (arguments.hip_target,)
            for requested_target in targets:
                bundle, text_size, records = hip_records(
                    hip_library, requested_target
                )
                target = bundle.rsplit("--", 1)[-1]
                hip_evidence.append(_target_profile_evidence(target, records))
                print(f"  bundle: {bundle}")
                if text_size is not None:
                    print(
                        f"  code-object .text: {text_size:,} bytes (0x{text_size:x})"
                    )
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
                if arguments.require_precision_profiles:
                    failures.extend(
                        precision_specialization_errors(records, f"HIP {bundle}")
                    )
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
        if arguments.write_evidence_dir is not None:
            assert evidence_wheel is not None
            if cuda_library is not None:
                _write_precision_evidence(
                    arguments.write_evidence_dir,
                    "cuda",
                    cuda_library,
                    evidence_wheel,
                    cuda_evidence,
                )
            if hip_library is not None:
                _write_precision_evidence(
                    arguments.write_evidence_dir,
                    "rocm",
                    hip_library,
                    evidence_wheel,
                    hip_evidence,
                )
        print("\nStatic artifact checks: PASS")
        return 0
    except ReportError as error:
        print(f"gpu_static_kernel_report.py: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
