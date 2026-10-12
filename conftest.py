"""Shared pytest fixtures for the project's test suite.

The central fixture, ``fake_dataset``, builds a tiny synthetic dataset on
disk that mirrors the folder layout of a Roboflow YOLO export
(``<split>/images``, ``<split>/labels``, ``data.yaml``). Using synthetic
data keeps the tests fast, offline, and independent of the real dataset,
while still containing the edge cases the real data is known to contain
(see Section 2 of the project documentation): pass images, repeated boxes
of one class, multi-defect images, a missing label file, and a corrupt
image file.
"""

from pathlib import Path

import pytest
import yaml
from PIL import Image

CLASS_NAMES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"]


def _write_image(path: Path, colour=(120, 80, 60)) -> None:
    """Writes a small valid JPEG image to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), colour).save(path, format="JPEG")


def _write_label(path: Path, lines: list[str]) -> None:
    """Writes a YOLO-format label file with the given lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Builds a small synthetic YOLO-format dataset and returns its root.

    Contents (class IDs: 0 loose-meat, 1 packaging-error, 2 twisted-meat,
    3 unsealed, 4 wrinkle):

    train/
        pass_a.jpg          empty label file            -> pass
        pass_no_label.jpg   no label file at all        -> pass
        unsealed_a.jpg      one class-3 box             -> fail, single defect
        pkg_two_boxes.jpg   two class-1 boxes           -> fail, single defect
                            (repeated boxes of ONE class are not "multi-defect")
        multi_a.jpg         class-1 and class-3 boxes   -> fail, multi-defect
        corrupt.jpg         not a real image            -> excluded entirely
    valid/
        pass_v.jpg          empty label file            -> pass
        loose_v.jpg         one class-0 box             -> fail, single defect
    test/
        pkg_t.jpg           one class-1 box             -> fail, single defect
    """
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"names": CLASS_NAMES, "nc": 5}))

    box = "{cls} 0.5 0.5 0.2 0.2"

    # --- train split ---
    _write_image(root / "train/images/pass_a.jpg")
    _write_label(root / "train/labels/pass_a.txt", [])

    _write_image(root / "train/images/pass_no_label.jpg")  # deliberately no label file

    _write_image(root / "train/images/unsealed_a.jpg")
    _write_label(root / "train/labels/unsealed_a.txt", [box.format(cls=3)])

    _write_image(root / "train/images/pkg_two_boxes.jpg")
    _write_label(root / "train/labels/pkg_two_boxes.txt", [box.format(cls=1), box.format(cls=1)])

    _write_image(root / "train/images/multi_a.jpg")
    _write_label(root / "train/labels/multi_a.txt", [box.format(cls=1), box.format(cls=3)])

    corrupt = root / "train/images/corrupt.jpg"
    corrupt.write_bytes(b"this is definitely not a jpeg")
    _write_label(root / "train/labels/corrupt.txt", [box.format(cls=3)])

    # --- valid split ---
    _write_image(root / "valid/images/pass_v.jpg")
    _write_label(root / "valid/labels/pass_v.txt", [])
    _write_image(root / "valid/images/loose_v.jpg")
    _write_label(root / "valid/labels/loose_v.txt", [box.format(cls=0)])

    # --- test split ---
    _write_image(root / "test/images/pkg_t.jpg")
    _write_label(root / "test/labels/pkg_t.txt", [box.format(cls=1)])

    return root


@pytest.fixture
def class_names() -> list[str]:
    """The project's five defect-type class names, in class-ID order."""
    return list(CLASS_NAMES)
