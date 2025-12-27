"""Memory guard tests."""
from gafime.backend.memory_guard import VramInfo, should_keep_in_vram


def test_keep_in_vram_false_when_insufficient():
    info = VramInfo(free_bytes=1024, total_bytes=2048)
    assert should_keep_in_vram(True, required_bytes=2000, vram_info=info) is False


def test_keep_in_vram_true_when_sufficient():
    info = VramInfo(free_bytes=1024 * 1024, total_bytes=2 * 1024 * 1024)
    assert should_keep_in_vram(True, required_bytes=256, vram_info=info) is True
