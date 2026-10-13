import numpy as np

from r1 import boxes as bx


def test_mask_to_boxes_two_components_and_tiny_dropped():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:50] = 255      # component A: box (10,10)-(49,29)
    mask[60:70, 70:75] = 1        # component B: any value > 0 counts
    mask[90, 90] = 255            # single pixel: contour with 1 vertex, dropped
    boxes = bx.mask_to_boxes(mask)
    assert boxes.shape == (2, 4)
    boxes = boxes[np.argsort(boxes[:, 0])]
    assert boxes[0].tolist() == [10, 10, 49, 29]
    assert boxes[1].tolist() == [70, 60, 74, 69]


def test_yolo_roundtrip_matches_original_format():
    boxes = np.array([[10, 10, 49, 29]], dtype=float)
    lines = bx.boxes_to_yolo_lines(boxes, 640, 640)
    assert lines == ["0 0.046094 0.030469 0.060937 0.029687"]
    back = bx.yolo_to_xyxy(np.array([[float(v) for v in lines[0].split()]]), 640, 640)
    assert np.allclose(back, boxes, atol=0.01)


def test_iou_matrix_values():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=float)
    iou = bx.iou_matrix(a, b)
    assert np.allclose(iou, [[1.0, 25 / 175, 0.0]])
    assert bx.iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_size_buckets():
    assert bx.size_bucket(31 * 31) == "small"
    assert bx.size_bucket(50 * 50) == "medium"
    assert bx.size_bucket(100 * 100) == "large"
