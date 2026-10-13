import numpy as np

from revision import compare_protocols as cp
from revision import sam3_eval as se
from revision.check_consistency import matches
from revision.make_tables import Numbers


def test_pixel_metrics_handles_empty_and_partial_masks():
    gt = np.zeros((10, 10), dtype=bool)
    gt[2:6, 2:6] = True
    assert se.pixel_metrics(np.zeros_like(gt), gt) == (0.0, 0.0)          # undetected positive -> 0, never 1
    assert se.pixel_metrics(gt.copy(), gt) == (1.0, 1.0)
    pred = np.zeros_like(gt)
    pred[2:6, 2:4] = True                                                   # half of the object
    iou, dice = se.pixel_metrics(pred, gt)
    assert abs(iou - 0.5) < 1e-9 and abs(dice - 2 / 3) < 1e-9


def test_consistency_matching_respects_written_precision():
    values = {0.8196, 0.35, 228.0}
    assert matches("0.82", values) and matches("0.820", values) and not matches("0.819", values)
    assert matches("82", values) and matches("81.96", values) and not matches("83", values)   # percentages of a fraction
    assert matches("228", values) and not matches("229", values)
    assert not matches("0.5", values)


def test_numbers_formatting_and_recording():
    n = Numbers()
    assert n.add("t", 0.0) == "0.000" and n.add("t", 1.0) == "1.000" and n.add("t", 33) == "33" and n.add("t", 130.64) == "130.6"
    assert n.add("t", float("nan")) == "--" and n.add("t", "text") == "text"
    assert n.values["t"] == [0.0, 1.0, 33.0, 130.64]


def test_paired_protocol_rows():
    rows = []
    for tag, base in (("dtl", 0.30), ("stl", 0.33)):
        for seed in (0, 1):
            for fold in range(3):
                rows.append({"tag": tag, "model": "yolo26m", "seed": str(seed), "fold": str(fold), "map50": base + 0.01 * fold, "map5095": base / 2, "precision": 0.5, "recall": 0.5, "f1": 0.5})
    out = cp.paired_rows(rows, "val", "dtl", "stl")
    assert len(out) == 1 and out[0]["n_pairs"] == 6
    assert abs(out[0]["map50_delta"] - 0.03) < 1e-6 and out[0]["map50_wins"] == 6
    assert out[0]["f1_delta"] == 0.0
