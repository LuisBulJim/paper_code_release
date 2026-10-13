import importlib.util
from pathlib import Path

import numpy as np

from r1 import evaluate as ev
from r1 import nested_eval as ne
from r1 import predict_dump as pdump
from r1.sweep_inference import TTA_GRID


def random_set(seed: int, n_images: int = 40):
    rng = np.random.default_rng(seed)
    preds, gt, names = {}, {}, []
    for i in range(n_images):
        name = f"image_{i:04d}.jpg"
        names.append(name)
        n_gt = int(rng.integers(0, 4))
        xy = rng.uniform(0, 500, (n_gt, 2))
        g = np.hstack([xy, xy + rng.uniform(20, 120, (n_gt, 2))])
        gt[name] = {"boxes": g, "classes": np.zeros(n_gt, dtype=int), "width": 640, "height": 640}
        jitter = g + rng.normal(0, 12, g.shape)
        xy = rng.uniform(0, 500, (int(rng.integers(0, 5)), 2))
        noise = np.hstack([xy, xy + rng.uniform(10, 100, xy.shape)])
        b = np.vstack([jitter, noise]) if len(noise) else jitter
        s = np.round(rng.uniform(0.001, 1, len(b)), 3)  # rounded: ties exercise the stable sort
        preds[name] = {"boxes": b, "scores": s, "classes": np.zeros(len(b), dtype=int)}
    return preds, gt, names


def test_fast_coco_equals_pycocotools_on_full_set_and_resamples():
    preds, gt, names = random_set(0)
    per_image = ne.coco_per_image(preds, gt, names)
    for sample in (names, [names[i] for i in np.random.default_rng(1).integers(0, len(names), len(names))], names[:7] + names[:7]):
        if not any(len(gt[n]["boxes"]) for n in sample):
            continue
        ref = ev.coco_map(preds, gt, sample)
        fast = ne.coco_from_per_image(per_image, sample)
        assert abs(ref["map50"] - fast["map50"]) < 1e-12 and abs(ref["map5095"] - fast["map5095"]) < 1e-12


def test_fast_coco_handles_images_without_detections():
    preds, gt, names = random_set(2, 12)
    for n in names[:6]:
        preds[n] = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "classes": np.zeros(0, dtype=int)}
    per_image = ne.coco_per_image(preds, gt, names)
    ref = ev.coco_map(preds, gt, names)
    fast = ne.coco_from_per_image(per_image, names)
    assert abs(ref["map50"] - fast["map50"]) < 1e-12 and abs(ref["map5095"] - fast["map5095"]) < 1e-12


def test_view_keys_match_nested_cv():
    spec = importlib.util.spec_from_file_location("juan_common", Path(__file__).resolve().parents[2] / "common.py")
    juan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(juan)
    for s in (0.8, 0.9, 1.0, 1.1, 1.2):
        for flip in (False, True):
            assert ne.view_key(s, flip) == juan.view_key("imgsz", s, flip)


def test_tta_from_views_equals_predict_dump_fusion():
    rng = np.random.default_rng(3)
    cfg = TTA_GRID["tta_s3_flip"]
    views, parts = {}, []
    for key, weight in ne.tta_keys(cfg):
        xy = rng.uniform(0, 0.7, (3, 2))
        b = np.hstack([xy, xy + 0.2])
        s = rng.uniform(0.01, 1, 3)
        views[key] = (b.astype(np.float32), s.astype(np.float32))
        parts.append((b.astype(np.float32) * 640.0, s.astype(np.float32).astype(float), np.zeros(3, dtype=int), weight))
    got = ne.det_tta(views, cfg, 0.5, 0.001, 640, 640)
    exp_b, exp_s, _ = pdump.fuse(parts, 640, 640, 0.5, 0.001)
    assert np.allclose(got["boxes"], exp_b) and np.allclose(got["scores"], exp_s)
    assert set(ne.available_tta(views)) == {"tta_s3_flip", "tta_s3_noflip", "tta_flip_only", "tta_s3_flip_equal"}


def test_reconstructed_groups_are_contiguous_blocks_with_the_same_folds():
    names = [f"image_{i:04d}.jpg" for i in (1, 2, 3, 10, 11, 20)]
    splits = {
        "0": [{"test": names[:3]}, {"test": names[3:]}],
        "1": [{"test": names[:3] + names[5:]}, {"test": names[3:5]}],
    }
    g = ne.reconstruct_groups(splits)
    assert g[names[0]] == g[names[1]] == g[names[2]]
    assert g[names[3]] == g[names[4]] != g[names[0]]
    assert g[names[5]] not in (g[names[0]], g[names[3]])


def test_cluster_samples_keep_groups_whole_and_both_classes():
    names = [f"i{k}" for k in range(12)]
    groups = {n: f"g{k // 3}" for k, n in enumerate(names)}
    positive = {n: k < 6 for k, n in enumerate(names)}
    samples, _ = ne.cluster_samples(groups, names, positive, 200, 0)
    assert len(samples) == 200
    for s in samples:
        flags = [positive[n] for n in s]
        assert any(flags) and not all(flags)
        for g in {groups[n] for n in s}:
            assert sum(groups[n] == g for n in s) % 3 == 0


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(4)
    s = np.round(rng.uniform(0, 1, 50), 1)
    y = rng.uniform(0, 1, 50) < 0.6
    pairs = [(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]]
    assert abs(ne.auroc(s, y) - float(np.mean(pairs))) < 1e-12


def test_scored_metrics_match_evaluator():
    preds, gt, names = random_set(5)
    sc = ne.Scored(preds, gt, names, 0.25, 0.5)
    m = sc.metrics(names)
    ref = ev.all_metrics(preds, gt, names, 0.25, 0.5)
    for k in ("map50", "map5095", "precision", "recall", "f1"):
        assert abs(m[k] - ref[k]) < 1e-12
