import numpy as np

from r1.predict_dump import fuse


def _view(boxes, scores, weight=1.0, classes=None):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    classes = np.zeros(len(scores), dtype=int) if classes is None else np.asarray(classes, dtype=int)
    return boxes, scores, classes, weight


def test_fuse_single_view_returns_same_boxes():
    boxes = [[10, 20, 110, 220], [300, 300, 400, 380]]
    scores = [0.9, 0.4]
    fb, fs, fc = fuse([_view(boxes, scores)], 640, 640, iou_thr=0.5, skip_thr=0.001)
    order = np.argsort(-fs)
    assert np.allclose(fb[order], boxes, atol=0.05)
    assert np.allclose(fs[order], scores)
    assert fc.tolist() == [0, 0]


def test_fuse_merges_overlapping_views_and_down_weights_singletons():
    a = _view([[100, 100, 200, 200]], [0.8])
    b = _view([[102, 98, 202, 202], [400, 400, 450, 450]], [0.6, 0.9])
    fb, fs, _ = fuse([a, b], 640, 640, iou_thr=0.5, skip_thr=0.001)
    assert len(fb) == 2
    merged = fb[np.argmax(fb[:, 0] < 300)]
    assert 100 <= merged[0] <= 102 and 200 <= merged[2] <= 202
    # the box seen in only one of two views has its confidence halved by WBF (0.9 * 1/2)
    assert np.isclose(fs.min(), 0.45, atol=0.01)


def test_fuse_keeps_classes_apart():
    a = _view([[100, 100, 200, 200]], [0.8], classes=[0])
    b = _view([[100, 100, 200, 200]], [0.7], classes=[1])
    fb, fs, fc = fuse([a, b], 640, 640, iou_thr=0.5, skip_thr=0.001)
    assert len(fb) == 2 and sorted(fc.tolist()) == [0, 1]


def test_fuse_empty_views():
    fb, fs, fc = fuse([_view(np.zeros((0, 4)), [])], 640, 640, 0.5, 0.001)
    assert fb.shape == (0, 4) and fs.shape == (0,) and fc.shape == (0,)


def test_horizontal_flip_mapping():
    w = 640
    fb = np.array([[10, 50, 110, 150]], dtype=float)
    fb[:, [0, 2]] = w - fb[:, [2, 0]]
    assert fb.tolist() == [[530, 50, 630, 150]]
