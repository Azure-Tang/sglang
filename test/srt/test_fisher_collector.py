import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_SRC = REPO_ROOT / "sglang" / "python"
COLLECTOR_PATH = PY_SRC / "sglang" / "srt" / "models" / "fisher_collector.py"

MODULE_NAME = "test_fisher_collector_module"


def _ensure_cuda_graph_stub(monkeypatch):
    stub_cuda = types.ModuleType("sglang.srt.model_executor.cuda_graph_runner")
    stub_cuda.get_is_capture_mode = lambda: False

    stub_model_executor = types.ModuleType("sglang.srt.model_executor")
    stub_model_executor.cuda_graph_runner = stub_cuda

    stub_srt = types.ModuleType("sglang.srt")
    stub_srt.model_executor = stub_model_executor

    stub_sglang = types.ModuleType("sglang")
    stub_sglang.srt = stub_srt

    monkeypatch.setitem(sys.modules, "sglang", stub_sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", stub_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor", stub_model_executor)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.cuda_graph_runner",
        stub_cuda,
    )


def _reload_collector(monkeypatch, **env_vars):
    for key, value in env_vars.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    _ensure_cuda_graph_stub(monkeypatch)

    spec = importlib.util.spec_from_file_location(MODULE_NAME, COLLECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "get_is_capture_mode", lambda: False, raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False, raising=False)

    return module, module.FisherCollector()


def _expected_stats(router_logits, hidden_states, top_k):
    probs = torch.softmax(router_logits, dim=-1)
    x_norm = torch.norm(hidden_states, p=2, dim=-1)
    fisher = ((probs * x_norm.unsqueeze(-1)) ** 2).sum(dim=0)

    _, topk_indices = torch.topk(probs, k=top_k, dim=-1)
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    probs_drop = probs.clone()
    probs_drop[mask] = 0.0
    co_activation = probs_drop.T @ probs_drop
    return fisher, co_activation


def test_fisher_collector_accumulates_stats(monkeypatch):
    module, collector = _reload_collector(
        monkeypatch,
        ENABLE_FISHER_CALIBRATION="1",
        FISHER_SKIP_IN_CUDAGRAPH="0",
        FISHER_DEVICE_STAGE="0",
    )

    layer_id = 2
    router_logits = torch.tensor([[0.1, 0.3, 0.2], [0.0, 0.4, 0.6]], dtype=torch.float32)
    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    top_k = 1

    collector.update(layer_id, router_logits, hidden_states, top_k)

    expected_fisher, expected_co = _expected_stats(router_logits, hidden_states, top_k)
    torch.testing.assert_close(collector.fisher[layer_id], expected_fisher)
    torch.testing.assert_close(collector.co_activation[layer_id], expected_co)
    assert collector.count[layer_id] == router_logits.shape[0]


def test_fisher_collector_skips_during_cuda_graph(monkeypatch):
    module, collector = _reload_collector(
        monkeypatch,
        ENABLE_FISHER_CALIBRATION="1",
        FISHER_SKIP_IN_CUDAGRAPH="1",
        FISHER_DEVICE_STAGE="0",
    )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True, raising=False)

    collector.update(0, torch.zeros(1, 2), torch.zeros(1, 2), top_k=1)
    assert collector.fisher == {}
    assert collector.co_activation == {}
    assert collector.count == {}


def test_fisher_collector_flushes_device_stage(monkeypatch, tmp_path):
    save_path = tmp_path / "fisher_stats.pt"
    module, collector = _reload_collector(
        monkeypatch,
        ENABLE_FISHER_CALIBRATION="1",
        FISHER_SKIP_IN_CUDAGRAPH="0",
        FISHER_DEVICE_STAGE="1",
        FISHER_SAVE_PATH=str(save_path),
    )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True, raising=False)

    layer_id = 1
    router_logits = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float32)
    hidden_states = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)

    collector.update(layer_id, router_logits, hidden_states, top_k=1)

    assert collector.fisher == {}
    assert layer_id in collector.fisher_device

    saved = {}

    def fake_save(data, path):
        saved["data"] = data
        saved["path"] = path

    monkeypatch.setattr(torch, "save", fake_save, raising=False)

    collector.save()

    assert saved["path"].endswith(".rank0")
    expected_fisher, expected_co = _expected_stats(router_logits, hidden_states, top_k=1)
    torch.testing.assert_close(saved["data"]["fisher"][layer_id], expected_fisher)
    torch.testing.assert_close(saved["data"]["co_activation"][layer_id], expected_co)
    assert saved["data"]["count"][layer_id] == router_logits.shape[0]
