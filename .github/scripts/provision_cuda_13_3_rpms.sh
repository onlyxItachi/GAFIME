#!/usr/bin/env bash
# CI-only bootstrap: run inside cibuildwheel's Linux x86_64 manylinux container.
set -euo pipefail

fail() { printf 'CUDA bootstrap: %s\n' "$*" >&2; exit 1; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail "requires Linux x86_64"
[[ $# == 0 ]] || fail "does not accept arguments"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
manifest="$script_dir/cuda_13_3_rpms.sha256"
test -f "$manifest" || fail "missing RPM checksum manifest"
repository=https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64
key_sha256=27e46a2d43e125859fb8a62c3b75bf798aeb95fa6f7d9bf790c1167ed9a0b39c
expected_rpms=(
    cccl-13-3-13.3.3.4.1-1.x86_64.rpm
    cuda-crt-13-3-13.3.73-1.x86_64.rpm
    cuda-cudart-13-3-13.3.29-1.x86_64.rpm
    cuda-cudart-devel-13-3-13.3.29-1.x86_64.rpm
    cuda-culibos-devel-13-3-13.3.33-1.x86_64.rpm
    cuda-cuobjdump-13-3-13.3.73-1.x86_64.rpm
    cuda-nvcc-13-3-13.3.73-1.x86_64.rpm
    cuda-nvdisasm-13-3-13.3.73-1.x86_64.rpm
    cuda-toolkit-13-3-config-common-13.3.29-1.noarch.rpm
    cuda-toolkit-13-config-common-13.3.29-1.noarch.rpm
    cuda-toolkit-config-common-13.3.29-1.noarch.rpm
    libnvptxcompiler-13-3-13.3.73-1.x86_64.rpm
    libnvvm-13-3-13.3.73-1.x86_64.rpm
)
declare -A allowed=() seen=()
entry_count=0
for rpm_name in "${expected_rpms[@]}"; do allowed["$rpm_name"]=1; done
while read -r digest rpm_name extra || [[ -n "$digest" ]]; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ && "$rpm_name" =~ ^[a-z0-9_.-]+\.rpm$ && -z "$extra" ]] \
        || fail "malformed RPM checksum entry"
    [[ -n "${allowed[$rpm_name]:-}" && -z "${seen[$rpm_name]:-}" ]] \
        || fail "unexpected or duplicate RPM: $rpm_name"
    seen["$rpm_name"]=1
    ((entry_count += 1))
done < "$manifest"
[[ $entry_count == ${#expected_rpms[@]} ]] || fail "incomplete RPM checksum manifest"

cuda_download_dir=$(mktemp -d "${TMPDIR:-/tmp}/gafime-cuda-rpms.XXXXXX")
trap 'rm -rf -- "$cuda_download_dir"' EXIT
download() {
    curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
        --retry 3 --connect-timeout 30 --max-time 300 \
        --output "$cuda_download_dir/$1" "$repository/$1"
}
download D42D0685.pub
(cd "$cuda_download_dir" && printf '%s  D42D0685.pub\n' "$key_sha256" | sha256sum --check --strict)
cuda_rpm_paths=()
for rpm_name in "${expected_rpms[@]}"; do
    download "$rpm_name"
    cuda_rpm_paths+=("$cuda_download_dir/$rpm_name")
done
(cd "$cuda_download_dir" && sha256sum --check --strict "$manifest")

rpm --import "$cuda_download_dir/D42D0685.pub"
for cuda_rpm_path in "${cuda_rpm_paths[@]}"; do
    signature_report=$(rpm --checksig --verbose "$cuda_rpm_path")
    printf '%s\n' "$signature_report"
    grep -Eq 'Signature, key ID d42d0685: OK$' <<< "$signature_report" \
        || fail "missing verified NVIDIA signature: $cuda_rpm_path"
done
# Retain AlmaLinux repositories for gcc-c++ and OS dependencies. Never register
# NVIDIA's repository: its mutable metadata is not needed for these exact files.
dnf --disablerepo='cuda*' --setopt=localpkg_gpgcheck=1 install -y "${cuda_rpm_paths[@]}"

cuda_root=/usr/local/cuda-13.3
for component in bin/nvcc bin/ptxas bin/nvlink bin/cuobjdump bin/nvdisasm nvvm/bin/cicc; do
    test -x "$cuda_root/$component" || fail "missing executable: $component"
done
for component in include/cuda_runtime.h include/crt/host_config.h \
    nvvm/libdevice/libdevice.10.bc lib64/libcudart.so lib64/libcudart.so.13 lib64/libcudadevrt.a; do
    test -f "$cuda_root/$component" || fail "missing component: $component"
done
ln -sfn "$cuda_root" /usr/local/cuda
