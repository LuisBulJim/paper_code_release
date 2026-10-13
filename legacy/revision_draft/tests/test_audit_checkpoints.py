import pickle
import zipfile
from pathlib import Path

from revision.audit_checkpoints import audit


def fake_checkpoint(path: Path, data_yaml: Path) -> Path:
    payload = {"train_args": {"data": str(data_yaml), "model": "yolo26n.pt"}, "date": "2026-09-28T13:43:00", "layers": ["C2PSA", "SPPF"]}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", pickle.dumps(payload, protocol=2))
    return path


def make_run(tmp_path: Path, train_names: list[str]) -> Path:
    train_txt = tmp_path / "train.txt"
    train_txt.write_text("\n".join(str(tmp_path / "images" / "pool" / n) for n in train_names) + "\n")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(f"path: {tmp_path}\ntrain: {train_txt}\nval: {train_txt}\nnames:\n  0: corrosion\n")
    return fake_checkpoint(tmp_path / "best.pt", data_yaml)


def test_audit_flags_test_image_in_train_list(tmp_path):
    pt = make_run(tmp_path, ["image_0001.jpg", "image_0777.jpg"])
    report = audit(pt, {"image_0777.jpg", "image_0999.jpg"})
    assert report["data_yaml"] == [str(tmp_path / "data.yaml")]
    assert report["train_source_checked"] == str(tmp_path / "train.txt")
    assert report["test_images_in_train_list"] == ["image_0777.jpg"]
    assert report["is_yolo26"]


def test_audit_clean_run_has_no_overlap(tmp_path):
    pt = make_run(tmp_path, ["image_0001.jpg", "image_0002.jpg"])
    report = audit(pt, {"image_0777.jpg"})
    assert report["train_source_checked"] is not None
    assert report["test_images_in_train_list"] == []
