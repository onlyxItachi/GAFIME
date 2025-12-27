from gafime.backend.memory_guard import plan_gpu_batches, query_vram


def test_memory_guard_decision():
    status = query_vram()
    decision = plan_gpu_batches(
        total_rows=100,
        bytes_per_row=64,
        keep_in_vram=True,
    )
    if status.free_bytes is None:
        assert decision.use_gpu is False
        assert decision.reason
    else:
        assert decision.batch_rows >= 0
        assert decision.batch_rows <= 100
