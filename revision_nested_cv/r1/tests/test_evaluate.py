import numpy as np
import pytest

from r1 import evaluate as ev


def _gt(boxes_by_name):
    return {n: {"boxes": np.asarray(b, dtype=float).reshape(-1, 4), "width": 640, "height": 640} for n, b in boxes_by_name.items()}


def _pred(boxes, scores):
    return {"boxes": np.asarray(boxes, dtype=float).reshape(-1, 4), "scores": np.asarray(scores, dtype=float)}


GT = _gt({"a.jpg": [[10, 10, 100, 100], [200, 200, 300, 320]], "b.jpg": [[50, 50, 150, 150]], "neg.jpg": []})
NAMES = sorted(GT)


def test_perfect_predictions_score_one():
    preds = {n: _pred(GT[n]["boxes"], [0.9] * len(GT[n]["boxes"])) for n in NAMES}
    m = ev.all_metrics(preds, GT, NAMES, conf=0.25, iou_thr=0.5)
    assert m["map50"] == pytest.approx(1.0) and m["map5095"] == pytest.approx(1.0)
    assert (m["precision"], m["recall"], m["f1"]) == (1.0, 1.0, 1.0)
    assert m["n_gt_boxes"] == 3 and m["tp"] == 3


def test_no_predictions_score_zero():
    preds = {n: _pred(np.zeros((0, 4)), []) for n in NAMES}
    m = ev.all_metrics(preds, GT, NAMES, conf=0.25, iou_thr=0.5)
    assert m["map50"] == 0.0 and m["recall"] == 0.0 and m["precision"] == 0.0 and m["fn"] == 3


def test_fixed_point_counts_fp_on_negative_and_respects_confidence():
    preds = {
        "a.jpg": _pred([[10, 10, 100, 100], [200, 200, 300, 320]], [0.9, 0.1]),   # second box below conf 0.25 -> FN
        "b.jpg": _pred([[50, 50, 150, 150]], [0.8]),
        "neg.jpg": _pred([[0, 0, 40, 40]], [0.7]),                                 # FP on a background image
    }
    m = ev.fixed_point_metrics(preds, GT, NAMES, conf=0.25, iou_thr=0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)


def test_shifted_box_below_iou_threshold_is_a_miss():
    preds = {"a.jpg": _pred([[60, 60, 150, 150]], [0.9]), "b.jpg": _pred(np.zeros((0, 4)), []), "neg.jpg": _pred(np.zeros((0, 4)), [])}
    m = ev.fixed_point_metrics(preds, GT, NAMES, conf=0.25, iou_thr=0.5)
    assert m["tp"] == 0 and m["fp"] == 1


def test_bootstrap_and_paired_delta_are_deterministic():
    good = {n: _pred(GT[n]["boxes"], [0.9] * len(GT[n]["boxes"])) for n in NAMES}
    bad = {n: _pred(np.zeros((0, 4)), []) for n in NAMES}
    ci1 = ev.bootstrap_ci(lambda sub: ev.all_metrics(good, GT, sub, 0.25, 0.5), NAMES, GT, iters=20, seed=3)
    ci2 = ev.bootstrap_ci(lambda sub: ev.all_metrics(good, GT, sub, 0.25, 0.5), NAMES, GT, iters=20, seed=3)
    assert ci1 == ci2 and ci1["map50"]["ci95"] == [1.0, 1.0]
    delta = ev.paired_bootstrap(bad, good, GT, NAMES, 0.25, 0.5, iters=20, seed=3)
    assert delta["map50"]["delta_mean"] == pytest.approx(1.0) and delta["map50"]["p_le0"] == 0.0


def test_bootstrap_samples_are_stratified():
    reps = ev.bootstrap_samples(NAMES, GT, iters=50, seed=0)
    assert all(len(r) == 3 and r.count("neg.jpg") == 1 for r in reps)


def test_class_aware_matching_and_per_class_metrics():
    gt = {
        "a.jpg": {"boxes": np.array([[10, 10, 100, 100], [200, 200, 300, 300]], dtype=float), "classes": np.array([0, 1]), "width": 640, "height": 640},
        "b.jpg": {"boxes": np.array([[50, 50, 150, 150]], dtype=float), "classes": np.array([1]), "width": 640, "height": 640},
    }
    preds = {
        "a.jpg": {"boxes": np.array([[10, 10, 100, 100], [200, 200, 300, 300]], dtype=float), "scores": np.array([0.9, 0.9]), "classes": np.array([0, 0])},  # second box has the wrong class
        "b.jpg": {"boxes": np.array([[50, 50, 150, 150]], dtype=float), "scores": np.array([0.9]), "classes": np.array([1])},
    }
    names = sorted(gt)
    m = ev.all_metrics(preds, gt, names, 0.25, 0.5, class_names={0: "Metal", 1: "Bio"})
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert m["classes"]["Metal"]["tp"] == 1 and m["classes"]["Metal"]["fp"] == 1
    assert m["classes"]["Bio"]["tp"] == 1 and m["classes"]["Bio"]["fn"] == 1
    assert m["classes"]["Metal"]["map50"] == pytest.approx(1.0)
    assert 0.0 < m["classes"]["Bio"]["map50"] < 1.0
    assert 0.0 < m["map50"] < 1.0


def test_match_details_reports_indices():
    pred = np.array([[10, 10, 100, 100], [500, 500, 600, 600], [12, 12, 98, 98]], dtype=float)
    scores = np.array([0.5, 0.9, 0.95])
    gt = np.array([[10, 10, 100, 100]], dtype=float)
    d = ev.match_details(pred, scores, gt, 0.25, 0.5)
    assert d["pred_idx"].tolist() == [2, 1, 0]          # score order
    assert d["pred_match"].tolist() == [0, -1, -1]      # best-scoring overlap wins, the duplicate becomes a FP
    assert d["gt_found"].tolist() == [True]


def test_single_class_records_without_classes_still_work():
    preds = {n: _pred(GT[n]["boxes"], [0.9] * len(GT[n]["boxes"])) for n in NAMES}
    m = ev.all_metrics(preds, GT, NAMES, conf=0.25, iou_thr=0.5)
    assert m["tp"] == 3 and m["map50"] == pytest.approx(1.0)
