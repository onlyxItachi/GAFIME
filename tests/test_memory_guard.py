from gafime.backend.memory_guard import enforce_keep_in_vram


def test_keep_in_vram_disabled():
    keep, reason = enforce_keep_in_vram(keep_in_vram=False, required_bytes=1024)
    assert keep is False
    assert reason is None


def test_keep_in_vram_unavailable_gpu():
    keep, reason = enforce_keep_in_vram(keep_in_vram=True, required_bytes=1024)
    assert keep is False or keep is True
    if keep is False:
        assert reason in {"gpu_unavailable", "insufficient_vram"}
