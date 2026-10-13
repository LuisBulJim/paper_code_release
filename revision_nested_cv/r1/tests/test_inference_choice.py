import logging

from r1 import predict_dump as pdump
from r1 import sweep_inference as sw
from r1.protocol import load_protocol, write_json


def _row(group, config, iou, model, map50, conf=0.25, weights=""):
    return {"group": group, "config": config, "wbf_iou": iou, "weights": weights, "model": model, "conf": conf, "map50_mean": map50}


def test_tta_choice_averages_over_models_and_ignores_standard():
    summary = [
        _row("tta", "standard", None, "n", 0.90), _row("tta", "standard", None, "m", 0.90),
        _row("tta", "tta_s3_flip", 0.5, "n", 0.30), _row("tta", "tta_s3_flip", 0.5, "m", 0.50),     # mean 0.40
        _row("tta", "tta_s5_flip", 0.4, "n", 0.45), _row("tta", "tta_s5_flip", 0.4, "m", 0.41),     # mean 0.43 -> chosen
        _row("tta", "tta_s5_flip", 0.4, "n", 0.99, conf=0.10), _row("tta", "tta_s5_flip", 0.6, "n", 0.99),  # other conf / incomplete
    ]
    c = sw.choose_tta(summary, 0.25)
    assert (c["config"], c["wbf_iou"]) == ("tta_s5_flip", 0.4)
    assert c["val_map50_mean_over_models"] == 0.43 and c["standard_val_map50_mean_over_models"] == 0.9


def test_wbf_choice_keeps_members_weights_and_iou():
    summary = [_row("wbf", "n+m_equal", 0.5, "n+m", 0.40), _row("wbf", "n+m_favour_larger", 0.6, "n+m", 0.42, weights="1,2")]
    assert sw.choose_wbf(summary, 0.25) == {"models": ["n", "m"], "weights": [1.0, 2.0], "iou": 0.6, "val_map50_mean": 0.42, "config": "n+m_favour_larger"}
    assert sw.choose_wbf([], 0.25) is None


def test_dump_is_reused_only_with_identical_settings(tmp_path):
    proto = load_protocol()
    images = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    ckpt = tmp_path / "best.pt"
    out = tmp_path / "dump"
    out.mkdir()
    for p in images:
        write_json(out / f"{p.stem}.json", {"image": p.name, "boxes": [], "scores": []})
    local = pdump.tta_protocol(proto, sw.TTA_GRID["tta_s3_flip"], 0.5)
    write_json(out / "_meta.json", {"strategy": "tta", "checkpoints": [{"path": str(ckpt), "sha256": "x"}], "model_weights": [1.0], "conf_dump": proto["evaluation"]["conf_dump"], "imgsz": proto["inference"]["imgsz"], "tta": local["inference"]["tta"], "wbf_models": None, "n_images": 2})
    assert pdump.dump_matches(out, "tta", [ckpt], images, local, None)
    assert not pdump.dump_matches(out, "tta", [ckpt], images, pdump.tta_protocol(proto, sw.TTA_GRID["tta_s3_flip"], 0.4), None)
    assert not pdump.dump_matches(out, "tta", [ckpt], images, pdump.tta_protocol(proto, sw.TTA_GRID["tta_s5_flip"], 0.5), None)
    assert not pdump.dump_matches(out, "tta", [tmp_path / "other.pt"], images, local, None)
    assert not pdump.dump_matches(out, "tta", [ckpt], images + [tmp_path / "c.jpg"], local, None)
    assert pdump.ensure_dump("tta", [ckpt], images, out, local, "cpu", None, logging.getLogger("t")) == out   # no model needed when reused
