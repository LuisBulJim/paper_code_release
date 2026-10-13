"""Train one YOLO26 variant with K-fold cross-validation on the training pool, one run per (model, seed).

For every fold of the requested seed:
  * data yaml built from image lists (no file copies); train = fold train, val = fold val; test never appears
  * YOLO.train(**hparams.yaml, seed=seed); best.pt is selected by Ultralytics fitness on the fold's val split
  * predictions of best.pt are dumped (standard strategy, conf 0.001) for the val fold AND the test set, so
    later stages never need the checkpoints again; test dumps are NOT evaluated here (collect_runs.py does it)
  * metrics on val (evaluate.py), run_info.json with hyperparameters, environment and best epoch, then DONE
Re-running skips folds that already have DONE, so a killed job can be resumed with the same command.

Dry run (CPU, minutes): --dry-run trains 1 epoch on 8 train / 4 val images of fold 0 at imgsz 320, batch 4.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import yaml

from revision import evaluate as ev
from revision import predict_dump as pdump
from revision.protocol import (
    REPO_ROOT,
    check_environment,
    get_logger,
    load_hparams,
    load_protocol,
    read_json,
    resolve,
    seed_everything,
    sha256_file,
    write_json,
)

CLASS_NAMES = {0: "corrosion"}


def write_list(path: Path, images_dir: Path, names: list[str]) -> Path:
    path.write_text("\n".join(str((images_dir / n).resolve()) for n in names) + "\n", encoding="utf-8")
    return path


def write_data_yaml(run_dir: Path, dataset_root: Path, train: list[str], val: list[str]) -> Path:
    train_txt = write_list(run_dir / "train.txt", dataset_root / "images" / "pool", train)
    val_txt = write_list(run_dir / "val.txt", dataset_root / "images" / "pool", val)
    data = {"path": str(dataset_root.resolve()), "train": str(train_txt.resolve()), "val": str(val_txt.resolve()), "names": CLASS_NAMES}
    yaml_path = run_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def best_epoch_from_results(results_csv: Path) -> dict:
    """Ultralytics fitness = 0.1*mAP50 + 0.9*mAP50-95 on the val split; best.pt is the epoch with max fitness."""
    with open(results_csv, encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return {}
    fitness = [0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"]) for r in rows]
    k = max(range(len(rows)), key=lambda i: fitness[i])
    return {
        "epochs_run": len(rows),
        "best_epoch": int(float(rows[k]["epoch"])),
        "best_fitness": fitness[k],
        "best_val_map50_ultralytics": float(rows[k]["metrics/mAP50(B)"]),
        "best_val_map5095_ultralytics": float(rows[k]["metrics/mAP50-95(B)"]),
    }


def train_fold(model_name: str, seed: int, fold: dict, run_dir: Path, dataset_root: Path, test_names: list[str], hparams: dict, proto: dict, device, init_weights: Path | None, dry_run: bool, log, provenance: dict | None = None) -> dict:
    from ultralytics import YOLO

    run_dir.mkdir(parents=True, exist_ok=True)
    train_names, val_names = fold["train"], fold["val"]
    if dry_run:
        train_names, val_names = train_names[:8], val_names[:4]
    data_yaml = write_data_yaml(run_dir, dataset_root, train_names, val_names)

    seed_everything(seed)
    weights = str(init_weights) if init_weights else f"{model_name}.pt"
    model = YOLO(weights)
    train_kwargs = dict(hparams)
    train_kwargs.update({"data": str(data_yaml), "project": str(run_dir.parent), "name": run_dir.name, "exist_ok": True, "seed": seed, "device": device, "plots": False, "verbose": False})
    if dry_run:
        train_kwargs.update({"epochs": 1, "imgsz": 320, "batch": 4, "workers": 0, "amp": False})
    t0 = time.perf_counter()
    model.train(**train_kwargs)
    train_s = time.perf_counter() - t0

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"best.pt missing after training: {best}")
    best_model = [best]
    pdump.run("standard", best_model, [dataset_root / "images" / "pool" / n for n in val_names], run_dir / "preds_val", proto, device, None, log)
    pdump.run("standard", best_model, [dataset_root / "images" / "test" / n for n in test_names], run_dir / "preds_test", proto, device, None, log)
    val_metrics = ev.evaluate_dump(run_dir / "preds_val", dataset_root, "pool", proto, image_names=val_names)["metrics"]
    write_json(run_dir / "metrics_val.json", val_metrics)

    info = {
        "model": model_name,
        "seed": seed,
        "fold": fold["fold"],
        "init_weights": weights,
        "init_weights_sha256": sha256_file(Path(weights)) if Path(weights).exists() else None,
        "n_train": len(train_names),
        "n_val": len(val_names),
        "dry_run": dry_run,
        "hparams": {k: v for k, v in train_kwargs.items() if k not in ("data", "project", "name", "exist_ok", "verbose", "plots")},
        "train_seconds": train_s,
        "best_pt_sha256": sha256_file(best),
        "val_metrics": val_metrics,
        "environment": check_environment(proto),
        "provenance": provenance or {},
        **best_epoch_from_results(run_dir / "results.csv"),
    }
    write_json(run_dir / "run_info.json", info)
    (run_dir / "weights" / "last.pt").unlink(missing_ok=True)
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log.info(f"fold {fold['fold']} done in {train_s/60:.1f} min | best epoch {info.get('best_epoch')} | val mAP50 {val_metrics['map50']:.4f} F1 {val_metrics['f1']:.4f}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="yolo26n | yolo26m | yolo26l")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="dtl", help="experiment tag: dtl | stl | ...")
    ap.add_argument("--out", default="runs", help="output root (runs/<tag>/<model>_seed<seed>/fold<k>)")
    ap.add_argument("--init-weights", default=None, help="checkpoint to start from (sequential transfer); default COCO weights")
    ap.add_argument("--folds", default=None, help="folds json (default: protocol splits.folds_file)")
    ap.add_argument("--only-folds", default=None, help="comma-separated fold indices to run (default: all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict-env", action="store_true", help="fail if ultralytics/torch differ from protocol.yaml")
    args = ap.parse_args()

    proto = load_protocol()
    hparams = load_hparams()
    check_environment(proto, strict=args.strict_env)
    dataset_root = resolve(proto["dataset"]["output_root"])
    folds_file = resolve(args.folds or proto["splits"]["folds_file"])
    folds_doc = read_json(folds_file)
    if str(args.seed) not in folds_doc["seeds"]:
        raise SystemExit(f"seed {args.seed} not in {folds_file} (available: {list(folds_doc['seeds'])})")
    manifest = read_json(resolve(proto["dataset"]["manifest_file"]))
    if manifest["sha256"] != folds_doc["manifest_sha256"]:
        raise SystemExit("folds file was generated from a different manifest; rerun make_splits")

    device = args.device
    if device is None:
        import torch

        device = "0" if torch.cuda.is_available() else "cpu"
    out_root = resolve(args.out) / args.tag / f"{args.model}_seed{args.seed}" if not args.dry_run else resolve(args.out) / "dryrun" / f"{args.model}_seed{args.seed}"
    log = get_logger(f"train_{args.tag}_{args.model}_s{args.seed}", out_root / "train.log")
    log.info(f"device={device} folds={folds_file.name} sha={folds_doc['manifest_sha256'][:12]} hparams={json.dumps(hparams)}")

    selected = folds_doc["seeds"][str(args.seed)]
    if args.only_folds:
        wanted = {int(x) for x in args.only_folds.split(",")}
        selected = [f for f in selected if f["fold"] in wanted]
    test_names = folds_doc["test"]
    if args.dry_run:
        selected = selected[:1]
        positive = {e["image"] for e in manifest["images"] if e["positive"]}
        test_names = [n for n in test_names if n in positive][:4] + [n for n in test_names if n not in positive][:2]
        write_json(out_root / "dryrun_test_images.json", {"images": test_names})
    for fold in selected:
        run_dir = out_root / f"fold{fold['fold']}"
        if (run_dir / "DONE").exists():
            log.info(f"fold {fold['fold']} already done, skipping")
            continue
        provenance = {"folds_file": str(folds_file), "folds_file_sha256": sha256_file(folds_file), "manifest_sha256": manifest["sha256"], "tag": args.tag}
        train_fold(args.model, args.seed, fold, run_dir, dataset_root, test_names, hparams, proto, device, Path(args.init_weights) if args.init_weights else None, args.dry_run, log, provenance)
    log.info("all requested folds finished")


if __name__ == "__main__":
    main()
