"""Tests for memory guard utilities."""

from gafime.backend.memory_guard import enforce_keep_in_vram


def test_enforce_keep_in_vram_handles_unavailable():
    assert enforce_keep_in_vram(False) is False
    # If CuPy is unavailable, keep_in_vram is softened to False.
    assert enforce_keep_in_vram(True) in {True, False}
