"""Sanity checks for the memory-lean shared-embedding engine.

Run with:  python -m pytest tests -q      (or python tests/test_engine.py)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine import (  # noqa: E402
    IGNORE_INDEX,
    MAX_VOCAB_CHUNK,
    TOLERANCE,
    ChunkedSoftmaxLoss,
    ValidationError,
    build_conflict_report,
    decompose_shared_gradient,
    evaluate,
    make_weight,
    parse_payload,
    run_chunked,
)


def _payload(tokens, seed=7, scale=1.0, vocab=256):
    return {
        "seed": seed,
        "scale": scale,
        "vocab_size": vocab,
        "microbatches": [{"name": "mb", "tokens": [list(p) for p in tokens]}],
    }


def test_deterministic_weight_from_seed():
    a = make_weight(256, 1234, 1.0)
    b = make_weight(256, 1234, 1.0)
    c = make_weight(256, 1235, 1.0)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.dtype == torch.float64 and a.shape == (256, 16)
    scaled = make_weight(256, 1234, 2.5)
    assert torch.allclose(scaled, a * 2.5, rtol=0, atol=0)


def test_weight_tying_is_shared():
    spec = parse_payload(_payload([[3, 5], [7, 9]]))
    res = run_chunked(spec)
    # Gradient must include the input-embedding contribution as well as the
    # output-projection contribution.  Compare against a manual autograd graph
    # that ties the same tensor in both roles.
    w = make_weight(spec.vocab_size, spec.seed, spec.scale).requires_grad_(True)
    idx = torch.tensor([3, 7])
    tgt = torch.tensor([5, 9])
    hidden = w.index_select(0, idx)
    logits = hidden @ w.t()
    loss = -torch.log_softmax(logits, dim=1).gather(1, tgt[:, None]).mean()
    loss.backward()
    assert abs(float(loss.detach()) - res.loss) < 1e-12
    assert float((w.grad - res.grad).abs().max()) < 1e-12


def test_matches_dense_reference_with_ignored_targets():
    tokens = [[1, 2], [3, IGNORE_INDEX], [4, 5], [1, 6]]
    report = evaluate(_payload(tokens))
    assert report["ok"], report["errors"]
    assert report["errors"]["loss_abs_diff"] <= TOLERANCE
    assert report["errors"]["grad_max_abs_diff"] <= TOLERANCE
    assert report["errors"]["weight_max_abs_diff"] <= TOLERANCE


def test_microbatch_split_invariance():
    tokens = [[1, 2], [3, 4], [5, IGNORE_INDEX], [7, 8], [2, 2], [9, 10], [11, 12]]
    report = evaluate(_payload(tokens, seed=99, scale=3.0, vocab=512))
    assert report["ok"]
    assert report["split"]["max_loss_diff"] <= TOLERANCE
    assert report["split"]["max_grad_diff"] <= TOLERANCE
    assert report["split"]["max_weight_diff"] <= TOLERANCE
    assert len(report["split"]["runs"]) == 4


def test_sgd_update_matches_manual_step():
    spec = parse_payload(_payload([[1, 2], [3, 4], [5, 6]], seed=11, scale=0.7))
    res = run_chunked(spec)
    expected = res.base_weight - 0.01 * res.grad
    assert float((res.updated_weight - expected).abs().max()) == 0.0


def test_all_ignored_rejected_without_update():
    with pytest.raises(ValidationError):
        parse_payload(_payload([[1, IGNORE_INDEX], [2, IGNORE_INDEX]]))
    # Also via the Function directly: no update should be possible.
    w = make_weight(256, 1, 1.0).requires_grad_(True)
    hidden = w.index_select(0, torch.tensor([1, 2]))
    with pytest.raises(ValidationError):
        ChunkedSoftmaxLoss.apply(w, hidden, torch.tensor([IGNORE_INDEX, IGNORE_INDEX]))
    assert w.grad is None


def test_boundaries_and_validation():
    parse_payload({"seed": 0, "scale": 0.1, "vocab_size": 256,
                   "microbatches": [{"tokens": [[0, 255]]}]})
    parse_payload({"seed": 2**63 - 1, "scale": 100.0, "vocab_size": 4096,
                   "microbatches": [{"tokens": [[4095, 0]]}]})
    bad = [
        {"seed": 0, "scale": 0.09, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 100.1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 255,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 4097,
         "microbatches": [{"tokens": [[1, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[256, 2]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 256]]}]},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}] * 5},
        {"seed": 0, "scale": 1, "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]] * 129}]},
        {"seed": 0, "scale": 1, "vocab_size": 256, "microbatches": []},
        {"seed": 0, "scale": float("inf"), "vocab_size": 256,
         "microbatches": [{"tokens": [[1, 2]]}]},
    ]
    for payload in bad:
        with pytest.raises(ValidationError):
            parse_payload(payload)
    with pytest.raises(ValidationError):
        parse_payload({"seed": 0, "scale": 1, "vocab_size": 256,
                       "microbatches": [{"tokens": [[1, 2]]}], "bogus": 1})
    with pytest.raises(ValidationError):
        parse_payload({"seed": "abc", "scale": 1, "vocab_size": 256,
                       "microbatches": [{"tokens": [[1, 2]]}]})
    with pytest.raises(ValidationError):
        parse_payload({"seed": 1.5, "scale": 1, "vocab_size": 256,
                       "microbatches": [{"tokens": [[1, 2]]}]})


def test_token_budget_counts_whole_payload():
    # 4 micro-batches x 32 tokens = 128, allowed; 129 rejected.
    mbs = [{"name": str(i), "tokens": [[i, i + 1]] * 32} for i in range(4)]
    parse_payload({"seed": 0, "scale": 1, "vocab_size": 4096, "microbatches": mbs})
    mbs[3]["tokens"] = mbs[3]["tokens"] + [[0, 1]]
    with pytest.raises(ValidationError):
        parse_payload({"seed": 0, "scale": 1, "vocab_size": 4096, "microbatches": mbs})


@pytest.mark.parametrize("scale", [0.1, 1.0, 25.0, 100.0])
def test_large_scores_remain_accurate(scale):
    tokens = [[1, 4095], [2048, 0], [4095, 4095], [17, 33], [999, IGNORE_INDEX]]
    report = evaluate(_payload(tokens, seed=424242, scale=scale, vocab=4096))
    assert report["ok"], report["errors"]
    for key in ("loss_abs_diff", "grad_max_abs_diff", "weight_max_abs_diff"):
        assert report["errors"][key] <= TOLERANCE


def test_duplicate_inputs_accumulate_gradients():
    tokens = [[4, 4], [4, 4], [4, 7], [9, 4]]
    report = evaluate(_payload(tokens, seed=3, vocab=300))
    assert report["ok"]
    # Gradient rows for ids 4 and 9 must be non-trivially populated.
    spec = parse_payload(_payload(tokens, seed=3, vocab=300))
    res = run_chunked(spec)
    assert float(res.grad[4].abs().sum()) > 0
    assert float(res.grad[9].abs().sum()) > 0


def test_no_scores_or_probabilities_saved_across_forward_backward():
    """Record every tensor handed to save_for_backward during the run."""
    saved = []

    def pack(tensor):
        saved.append(tensor.detach().clone())
        return tensor

    def unpack(tensor):
        return tensor

    spec = parse_payload(_payload([[1, 2], [3, 4], [5, 6]], vocab=1024))
    weight = make_weight(spec.vocab_size, spec.seed, spec.scale).requires_grad_(True)
    hidden = weight.index_select(0, torch.tensor([1, 3, 5]))
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        loss = ChunkedSoftmaxLoss.apply(weight, hidden, torch.tensor([2, 4, 6]))
    loss.backward()

    assert saved, "save_for_backward was not called"
    shape_set = {tuple(t.shape) for t in saved}
    # Only W (V x 16), hidden states (T x 16), targets (T), valid idx (T).
    assert (1024, 16) in shape_set
    assert (3, 16) in shape_set
    for tensor in saved:
        assert tensor.numel() <= weight.numel()
        assert tensor.dim() <= 2
        if tensor.dim() == 2:
            assert tensor.shape[1] == 16 or tensor.shape[0] == 1024

    spec2 = parse_payload(_payload([[1, 2], [3, 4], [5, 6]], vocab=4096))
    res = run_chunked(spec2)
    for shapes in res.saved_shapes:
        # (W, hidden, target, valid_idx) for each micro-batch.
        assert shapes[0] == [4096, 16]
        assert shapes[1][1] == 16
        assert shapes[2][0] == shapes[3][0]


def test_gradient_matches_finite_differences():
    spec = parse_payload(
        {
            "seed": 12,
            "scale": 1.3,
            "vocab_size": 256,
            "microbatches": [
                {"tokens": [[3, 4], [3, 4], [200, 201], [7, IGNORE_INDEX]]},
                {"tokens": [[9, 3], [3, 3]]},
            ],
        }
    )
    res = run_chunked(spec)
    weight = make_weight(256, 12, 1.3)
    inputs = [3, 3, 200, 7, 9, 3]
    targets = [4, 4, 201, IGNORE_INDEX, 3, 3]

    def loss_at(w):
        ww = w.detach().clone().requires_grad_(True)
        idx = torch.tensor(inputs)
        tgt = torch.tensor(targets)
        valid = tgt != IGNORE_INDEX
        hidden = ww.index_select(0, idx)
        logits = hidden @ ww.t()
        picked = torch.log_softmax(logits, dim=1).gather(
            1, tgt.clamp_min(0)[:, None]
        ).squeeze(1)
        return -picked[valid].mean().detach()

    eps = 1e-6
    for i, j in [(3, 0), (4, 5), (200, 3), (9, 9)]:
        plus = weight.clone()
        minus = weight.clone()
        plus[i, j] += eps
        minus[i, j] -= eps
        numeric = float((loss_at(plus) - loss_at(minus)) / (2 * eps))
        analytic = float(res.grad[i, j])
        assert abs(numeric - analytic) <= 1e-7, (i, j, numeric, analytic)


def test_chunk_axis_never_exceeds_128(monkeypatch):
    import app.engine as engine

    widths = []
    real_range = engine._chunk_ranges

    def spy(vocab_size):
        for start, end in real_range(vocab_size):
            widths.append(end - start)
            yield start, end

    monkeypatch.setattr(engine, "_chunk_ranges", spy)
    spec = parse_payload(_payload([[1, 2], [3, 4]], vocab=4096))
    run_chunked(spec)
    assert widths and max(widths) <= MAX_VOCAB_CHUNK


def test_numeric_loss_equals_manual_formula():
    spec = parse_payload(_payload([[0, 1], [2, IGNORE_INDEX], [3, 3]], seed=5, scale=2.0))
    res = run_chunked(spec)
    w = make_weight(spec.vocab_size, spec.seed, spec.scale)
    hidden = w[[0, 2, 3]]
    logits = hidden @ w.t()
    expected = -torch.log_softmax(logits, dim=1)[[0, 2], [1, 3]].mean()
    assert abs(float(expected) - res.loss) < 1e-12
    assert math.isfinite(res.loss)


# ---------------------------------------------------------------------------
# Tied-embedding gradient decomposition / conflict diagnostics
# ---------------------------------------------------------------------------
def _split_reference(spec):
    """Ground truth: two independent copies of the initial W, one per role."""
    weight = make_weight(spec.vocab_size, spec.seed, spec.scale)
    w_in = weight.clone().requires_grad_(True)
    w_out = weight.clone().requires_grad_(True)
    inputs, targets = [], []
    for mb in spec.microbatches:
        inputs.extend(mb.input_ids)
        targets.extend(mb.target_ids)
    idx = torch.tensor(inputs)
    tgt = torch.tensor(targets)
    valid = tgt != IGNORE_INDEX
    hidden = w_in.index_select(0, idx)
    logits = hidden @ w_out.t()
    picked = torch.log_softmax(logits, dim=1).gather(
        1, tgt.clamp_min(0)[:, None]
    ).squeeze(1)
    loss = -picked[valid].mean()
    loss.backward()
    return w_in.grad.detach(), w_out.grad.detach()


def test_decomposition_matches_two_tensor_reference():
    payload = {
        "seed": 12,
        "scale": 1.3,
        "vocab_size": 512,
        "microbatches": [
            {"tokens": [[3, 4], [3, 4], [200, 201], [7, IGNORE_INDEX]]},
            {"tokens": [[9, 3], [3, 3], [3, 4]]},
        ],
    }
    spec = parse_payload(payload)
    weight = make_weight(spec.vocab_size, spec.seed, spec.scale)
    decomposition = decompose_shared_gradient(spec, weight)
    ref_input, ref_output = _split_reference(spec)

    assert float((decomposition.input_grad - ref_input).abs().max()) <= 1e-12
    assert float((decomposition.output_grad - ref_output).abs().max()) <= 1e-12

    # The two parts must add up to the gradient actually used for the SGD step.
    res = run_chunked(spec)
    combined = decomposition.input_grad + decomposition.output_grad
    assert float((combined - res.grad).abs().max()) <= TOLERANCE


def test_ignored_targets_contribute_no_gradient_and_repeats_accumulate():
    spec = parse_payload(
        _payload([[4, 4], [4, 4], [9, IGNORE_INDEX], [4, 7], [7, 4]], seed=3, vocab=300)
    )
    weight = make_weight(spec.vocab_size, spec.seed, spec.scale)
    decomposition = decompose_shared_gradient(spec, weight)

    # The ignored token (input 9) is not a valid training example, so it never
    # enters either path: its input row gets no lookup gradient.  Row 9 may
    # still receive output-path gradient as a *vocabulary* entry from other
    # tokens' softmax, so only the input path is asserted to be exactly zero.
    assert float(decomposition.input_grad[9].abs().sum()) == 0.0
    assert int(decomposition.input_grad[9].abs().gt(0).sum()) == 0

    # Row 4 is looked up three times among valid tokens and must accumulate the
    # sum of those per-token hidden gradients (ignored token 2 excluded).
    inputs = [4, 4, 9, 4, 7]
    targets = [4, 4, IGNORE_INDEX, 7, 4]
    idx = torch.tensor(inputs)
    tgt = torch.tensor(targets)
    valid = tgt != IGNORE_INDEX
    hidden = weight.index_select(0, idx).requires_grad_(True)
    logits = hidden @ weight.t()
    picked = torch.log_softmax(logits, dim=1).gather(
        1, tgt.clamp_min(0)[:, None]
    ).squeeze(1)
    loss = -picked[valid].mean()
    loss.backward()
    expected_row4 = hidden.grad[valid & (idx == 4)].sum(dim=0)
    assert float((decomposition.input_grad[4] - expected_row4).abs().max()) <= 1e-12


def test_conflict_report_ranking_and_formulas():
    spec = parse_payload(_payload([[4, 4], [4, 4], [7, 4], [4, 7], [9, 9]], seed=3))
    res = run_chunked(spec)
    report = build_conflict_report(spec, res.base_weight, res.grad)

    assert report["available"] is True
    assert report["decomposition_max_abs_diff"] <= TOLERANCE
    assert 1 <= len(report["top_rows"]) <= 10
    rows = report["top_rows"]
    cancellations = [r["cancellation"] for r in rows]
    assert cancellations == sorted(cancellations, reverse=True)

    for item in rows:
        assert item["input_norm"] > 0 and item["output_norm"] > 0
        assert item["norm_sum"] == item["input_norm"] + item["output_norm"]
        expected_cancel = item["norm_sum"] - item["combined_norm"]
        assert abs(item["cancellation"] - expected_cancel) <= 1e-12
        assert abs(item["cancellation_ratio"] - expected_cancel / item["norm_sum"]) <= 1e-12
        # The combined norm is the norm of the true sum of both rows, so the
        # cancellation is non-negative up to rounding.
        assert item["cancellation"] >= -1e-12

    # Every row with both paths non-zero is counted as qualifying, not just the
    # reported top 10; rows with a zero path (e.g. untouched vocab rows, or the
    # input-only id 9) are excluded.
    decomposition = decompose_shared_gradient(spec, res.base_weight)
    in_nz = decomposition.input_grad.norm(dim=1) > 0
    out_nz = decomposition.output_grad.norm(dim=1) > 0
    expected_qualifying = int((in_nz & out_nz).sum().item())
    assert report["qualifying_rows"] == expected_qualifying
    assert report["qualifying_rows"] >= len(rows)
    assert report["rows_with_input_grad"] == int(in_nz.sum().item())
    assert report["rows_with_output_grad"] == int(out_nz.sum().item())
    reported = {item["row"] for item in rows}
    assert reported <= set(torch.nonzero(in_nz & out_nz).flatten().tolist())


def test_conflict_report_matches_independent_sgd_component():
    # With a single distinct input id, the input path for that row is exactly
    # dL/dh, i.e. its rows can be reconstructed from the output-path residual.
    payload = {
        "seed": 77,
        "scale": 2.0,
        "vocab_size": 256,
        "microbatches": [{"tokens": [[5, 1], [5, 2], [5, IGNORE_INDEX]]}],
    }
    report = evaluate(payload)
    assert report["ok"]
    conflict = report["conflict"]
    assert conflict["available"] is True
    assert conflict["rows_with_input_grad"] == 1
    assert conflict["top_rows"][0]["row"] == 5
    assert conflict["top_rows"][0]["cancellation_ratio"] >= 0.0


def test_conflict_report_is_cleared_when_decomposition_disagrees(monkeypatch):
    import app.engine as engine

    spec = parse_payload(_payload([[1, 2], [3, 4]]))
    res = run_chunked(spec)
    tampered = res.grad.clone()
    tampered[0, 0] += 1.0

    report = build_conflict_report(spec, res.base_weight, tampered)
    assert report["available"] is False
    assert report["top_rows"] == []
    assert "不一致" in report["reason"]
    assert report["decomposition_max_abs_diff"] > TOLERANCE

    # An exception inside the decomposition must also clear the diagnostics.
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(engine, "decompose_shared_gradient", boom)
    report = build_conflict_report(spec, res.base_weight, res.grad)
    assert report["available"] is False
    assert report["top_rows"] == []
    assert "计算失败" in report["reason"]

    # And the training comparison is unaffected by a broken diagnostic.
    monkeypatch.undo()
    good = evaluate(_payload([[1, 2], [3, 4]]))
    assert good["ok"] and good["conflict"]["available"]


def test_conflict_report_rejects_non_finite_gradients():
    spec = parse_payload(_payload([[1, 2], [3, 4]]))
    res = run_chunked(spec)
    nan_grad = res.grad.clone()
    nan_grad[2, 3] = math.nan
    report = build_conflict_report(spec, res.base_weight, nan_grad)
    assert report["available"] is False
    assert report["top_rows"] == []
    assert "非有限" in report["reason"]

    import app.engine as engine

    original = engine.decompose_shared_gradient

    def non_finite_decomposition(spec_, weight_):
        decomposition = original(spec_, weight_)
        decomposition.input_grad[1, 0] = math.inf
        return decomposition

    engine.decompose_shared_gradient = non_finite_decomposition
    try:
        report = build_conflict_report(spec, res.base_weight, res.grad)
    finally:
        engine.decompose_shared_gradient = original
    assert report["available"] is False
    assert report["top_rows"] == []
    assert "非有限" in report["reason"]


def test_decomposition_respects_vocab_chunk_cap(monkeypatch):
    import app.engine as engine

    widths = []
    real_range = engine._chunk_ranges

    def spy(vocab_size):
        for start, end in real_range(vocab_size):
            widths.append(end - start)
            yield start, end

    monkeypatch.setattr(engine, "_chunk_ranges", spy)
    spec = parse_payload(_payload([[1, 2], [3, 4], [4095, 4094]], vocab=4096))
    weight = make_weight(spec.vocab_size, spec.seed, spec.scale)
    decompose_shared_gradient(spec, weight)
    assert widths and max(widths) <= MAX_VOCAB_CHUNK


def test_conflict_report_present_in_evaluate_payload():
    report = evaluate(_payload([[4, 4], [4, 4], [7, 4], [4, 7], [9, -100]]))
    conflict = report["conflict"]
    assert conflict["available"] is True
    assert conflict["max_vocab_chunk"] == MAX_VOCAB_CHUNK
    for key in (
        "row",
        "input_norm",
        "output_norm",
        "combined_norm",
        "norm_sum",
        "cancellation",
        "cancellation_ratio",
    ):
        assert key in conflict["top_rows"][0]
    # JSON-serialisable end-to-end.
    import json

    json.dumps(report)



if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
