"""GPU regression checks for decode metadata (no model weights required)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from nano_vllm.engine.model_runner import ModelRunner
from nano_vllm.engine.sequence import Sequence
from nano_vllm.utils.context import get_context, reset_context


@pytest.mark.parametrize("length,blocks,slot", [(1, [2], 512), (257, [3, 1], 256)])
def test_decode_includes_current_token(length, blocks, slot):
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(kvcache_block_size=256)
    seq = Sequence([1] * length)
    seq.block_table = blocks
    try:
        runner.prepare_decode([seq])
        context = get_context()
        assert context.context_lens.cpu().tolist() == [length]
        assert runner.positions.cpu().tolist() == [length - 1]
        assert context.slot_mapping.cpu().tolist() == [slot]
    finally:
        reset_context()


def test_graph_padding_cannot_reuse_previous_kv_slots():
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(enforce_eager=False)
    runner.input_ids = torch.ones(3, dtype=torch.long, device="cuda")
    runner.positions = torch.ones(3, dtype=torch.long, device="cuda")
    runner.decode_slot_mapping = torch.tensor([10, 20, 30], dtype=torch.int32, device="cuda")
    runner.decode_context_lens = torch.tensor([2, 2, 2], dtype=torch.int32, device="cuda")
    runner.decode_block_tables = torch.zeros((3, 1), dtype=torch.int32, device="cuda")
    variables = {
        "input_ids": torch.zeros(4, dtype=torch.long, device="cuda"),
        "positions": torch.zeros(4, dtype=torch.long, device="cuda"),
        "slot_mapping": torch.full((4,), 999, dtype=torch.int32, device="cuda"),
        "context_lens": torch.full((4,), 99, dtype=torch.int32, device="cuda"),
        "block_tables": torch.full((4, 2), 99, dtype=torch.int32, device="cuda"),
    }
    runner.graph_vars = {4: variables}
    runner.graph_hidden = {4: torch.zeros((4, 2), device="cuda")}
    runner.model = SimpleNamespace(compute_logits=lambda x: x)

    def replay():
        assert variables["slot_mapping"].tolist() == [10, 20, 30, -1]
        assert variables["context_lens"].tolist() == [2, 2, 2, 1]
        assert variables["block_tables"].count_nonzero().item() == 0

    runner.graphs = {4: SimpleNamespace(replay=replay)}
    try:
        assert runner.run_model(False).shape == (3, 2)
    finally:
        reset_context()
