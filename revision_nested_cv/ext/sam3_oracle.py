"""SAM 3 on the LIACI evaluation set without the detector, to separate detector misses from mask quality.

  --mode gt_boxes   prompts = the ground-truth boxes of each image (same Sam3Segmenter call as r1/sam3_eval.py),
                    an upper bound for the box-prompted cascade; --mock-sam fills the boxes instead (box-fill bound)
  --mode text       prompts = concept phrases (default "corrosion" and "rust"), zero-shot, no boxes at all

Writes the r1 layout (<out>/masks/*.png, <out>/boxes/*.json, <out>/_meta.json), so the metrics come from
  python -m r1.sam3_eval metrics --preds <out> --out <results dir> --figure <png>

Usage (server, sam2 env with transformers >= 5, from revision_nested_cv/):
  python ext/sam3_oracle.py --mode gt_boxes --out ext/out/sam3_gt_boxes
  python ext/sam3_oracle.py --mode text --out ext/out/sam3_text
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from r1.protocol import environment_report, load_protocol, read_json, resolve, write_json  # noqa: E402
from r1.sam3_eval import Sam3Segmenter, mock_segmenter  # noqa: E402


class Sam3TextSegmenter(Sam3Segmenter):
    """Promptable concept segmentation from a noun phrase: union of every instance returned for each phrase."""

    def text(self, image_rgb: np.ndarray, phrases: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = image_rgb.shape[:2]
        mask, boxes, scores = np.zeros((h, w), dtype=bool), [], []
        for phrase in phrases:
            inputs = self.processor(images=Image.fromarray(image_rgb), text=phrase, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            res = self.processor.post_process_instance_segmentation(outputs, threshold=self.threshold, target_sizes=inputs.get("original_sizes").tolist())[0]
            m = res.get("masks", [])
            m = m.cpu().numpy() if self.torch.is_tensor(m) else np.asarray(m)
            if len(m):
                mask |= np.logical_or.reduce(m.reshape(-1, h, w).astype(bool))
            for key, dst in (("boxes", boxes), ("scores", scores)):
                v = res.get(key, [])
                v = v.cpu().numpy() if self.torch.is_tensor(v) else np.asarray(v)
                dst.extend(v.tolist())
        return mask, np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(scores, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["gt_boxes", "text"])
    ap.add_argument("--prompts", nargs="+", default=["corrosion", "rust"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mock-sam", action="store_true", help="gt_boxes: fill the boxes instead of calling SAM 3")
    ap.add_argument("--limit", type=int, default=None, help="first N positives + N negatives (pipeline checks)")
    args = ap.parse_args()

    proto = load_protocol()
    cfg = proto["sam3"]
    data_root = resolve(cfg["output_root"])
    entries = read_json(resolve(cfg["manifest_file"]))["images"]
    if args.limit:
        entries = [e for e in entries if e["positive"]][: args.limit] + [e for e in entries if not e["positive"]][: args.limit]
    out = resolve(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "boxes").mkdir(parents=True, exist_ok=True)
    device = args.device
    if device is None and not args.mock_sam:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.mock_sam:
        if args.mode != "gt_boxes":
            raise SystemExit("--mock-sam only applies to --mode gt_boxes")
        segmenter = None
    elif args.mode == "text":
        segmenter = Sam3TextSegmenter(cfg["model_id"], device, cfg["mask_threshold"])
    else:
        segmenter = Sam3Segmenter(cfg["model_id"], device, cfg["mask_threshold"])

    times = []
    for i, e in enumerate(entries, 1):
        rgb = cv2.cvtColor(cv2.imread(str(data_root / "images" / e["image"])), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        if args.mode == "gt_boxes":
            boxes = np.asarray(e["gt_boxes_xyxy"], dtype=float).reshape(-1, 4)
            scores = np.ones(len(boxes))
            mask = mock_segmenter(rgb, boxes) if args.mock_sam else segmenter(rgb, boxes)
        else:
            mask, boxes, scores = segmenter.text(rgb, args.prompts)
        times.append(time.perf_counter() - t0)
        stem = Path(e["image"]).stem
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / "masks" / f"{stem}.png")
        write_json(out / "boxes" / f"{stem}.json", {"image": e["image"], "boxes": np.round(boxes, 2).tolist(), "scores": np.round(scores, 5).tolist()})
        if i % 25 == 0 or i == len(entries):
            print(f"  {i}/{len(entries)} images")
    label = f"{args.mode}" + (f":{'|'.join(args.prompts)}" if args.mode == "text" else "")
    write_json(out / "_meta.json", {"checkpoint": f"none ({label})", "mode": args.mode, "prompts": args.prompts if args.mode == "text" else None,
                                    "tta": False, "sam_model": "mock" if args.mock_sam else cfg["model_id"], "mask_threshold": cfg["mask_threshold"],
                                    "n_images": len(entries), "device": device, "mean_ms_sam": float(np.mean(times) * 1000), "environment": environment_report()})
    print("written to", out)


if __name__ == "__main__":
    main()
