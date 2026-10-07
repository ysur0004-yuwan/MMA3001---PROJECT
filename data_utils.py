"""data_utils.py

Utilities for converting YOLO-format bounding-box annotations from the
"Pork Rasher Error (Packaging)" dataset into image-level labels for a
classification task (pass/fail and defect-type), and for building a
TensorFlow data pipeline from those labels.

Project: MMA3001 Individual Project — Automated Classification of Pork
Rasher Packaging Errors.

Docstring style: Google style, used consistently across this module and
the rest of the project's source code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import tensorflow as tf
import yaml
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Standard input size for the transfer-learning backbone (e.g. MobileNetV2).
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def load_class_names(dataset_path: Path) -> list[str]:
    """Loads the class ID -> class name mapping from the dataset's data.yaml.

    Args:
        dataset_path: Root folder of the downloaded Roboflow dataset
            (the folder containing data.yaml, train/, valid/, test/).

    Returns:
        A list of class names, ordered by class ID (index 0 = class ID 0).

    Raises:
        FileNotFoundError: If data.yaml does not exist at the expected path.
    """
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Expected data.yaml at {yaml_path}, but it was not found."
        )
    with open(yaml_path) as f:
        data_yaml = yaml.safe_load(f)
    return data_yaml["names"]


def parse_yolo_label_file(label_path: Path) -> list[int]:
    """Parses a single YOLO-format .txt label file into a list of class IDs.

    Each non-empty line in a YOLO label file has the format:
        "<class_id> <x_center> <y_center> <width> <height>"
    Only the class_id is needed to build image-level labels; the bounding
    box coordinates are not used, since this project performs image-level
    classification rather than object detection (see Section 1 of the
    project documentation for the rationale behind that scope decision).

    Args:
        label_path: Path to a single .txt annotation file.

    Returns:
        A list of integer class IDs, one per annotated bounding box in the
        image. An empty list means the image has no annotated defects
        (a "pass" image).
    """
    if not label_path.exists():
        # No label file at all is treated the same as an empty one: a pass image.
        # This matches how Roboflow exports images with zero annotations.
        return []

    with open(label_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    class_ids = []
    for line in lines:
        try:
            class_ids.append(int(line.split()[0]))
        except (ValueError, IndexError):
            # Malformed line (e.g. truncated write, corrupt export) — log and
            # skip rather than crashing the whole data audit over one line.
            logger.warning(f"Malformed annotation line skipped in {label_path}: '{line}'")
    return class_ids


def is_image_readable(image_path: Path) -> bool:
    """Checks whether an image file can actually be opened and decoded.

    Used to filter out corrupt or unreadable files during data loading, per
    the project's documented handling of invalid inputs (Section 2 of the
    project documentation).

    Args:
        image_path: Path to the image file to check.

    Returns:
        True if the image opens and verifies successfully, False otherwise
        (a warning is logged in the False case).
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError) as e:
        logger.warning(f"Unreadable image skipped: {image_path} ({e})")
        return False


def build_image_level_labels(dataset_path: Path, class_names: list[str]) -> pd.DataFrame:
    """Builds a DataFrame of image-level labels for every image in every split.

    For each image, this function:
      1. Parses its YOLO label file to get the set of annotated defect classes.
      2. Derives a pass/fail label: "fail" if one or more defects are
         annotated, "pass" otherwise.
      3. Derives an extension (defect-type) label, but ONLY for images with
         exactly one distinct defect class. Images with zero defects (pass)
         or more than one distinct defect class (multi-defect) get
         extension_label = None, since forcing a single label onto a
         genuinely multi-defect image would misrepresent it — see the
         project documentation (Section 2) for the policy rationale.
      4. Skips images that fail a basic readability check.

    Args:
        dataset_path: Root folder of the downloaded Roboflow dataset.
        class_names: Class ID -> name mapping, from load_class_names().

    Returns:
        A pandas DataFrame with one row per valid image, and columns:
        "image_path", "split", "pass_fail", "is_multi_defect",
        "extension_label".
    """
    records = []
    for split in ["train", "valid", "test"]:
        image_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"

        if not image_dir.exists():
            logger.warning(f"Split '{split}' not found at {image_dir}, skipping.")
            continue

        image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

        for image_path in image_paths:
            if not is_image_readable(image_path):
                continue  # corrupt/unreadable images are excluded entirely

            label_path = label_dir / f"{image_path.stem}.txt"
            class_ids = parse_yolo_label_file(label_path)
            unique_classes = sorted(set(class_ids))

            pass_fail = "fail" if len(class_ids) > 0 else "pass"
            is_multi_defect = len(unique_classes) > 1

            if pass_fail == "pass":
                extension_label = None
            elif is_multi_defect:
                extension_label = None  # excluded from extension training, per policy
            else:
                extension_label = class_names[unique_classes[0]]

            records.append({
                "image_path": str(image_path),
                "split": split,
                "pass_fail": pass_fail,
                "is_multi_defect": is_multi_defect,
                "extension_label": extension_label,
            })

    df = pd.DataFrame.from_records(records)
    logger.info(f"Built image-level labels for {len(df)} images across all splits.")
    return df


def summarise_labels(df: pd.DataFrame) -> None:
    """Prints a quick summary of the label DataFrame, for sanity-checking.

    This is a lightweight manual-inspection helper, not a substitute for the
    proper data audit — it exists so the output of build_image_level_labels()
    can be eyeballed quickly against the known audit numbers (e.g. 3,549
    total images, 710 pass / 2,839 fail) before proceeding to training.

    Args:
        df: The DataFrame returned by build_image_level_labels().
    """
    print("Images per split:")
    print(df["split"].value_counts())

    print("\nPass/fail counts (overall):")
    print(df["pass_fail"].value_counts())

    print("\nExtension label counts (excludes pass and multi-defect images):")
    print(df["extension_label"].value_counts(dropna=True))

    print(f"\nMulti-defect images excluded from extension task: {int(df['is_multi_defect'].sum())}")


def make_pass_fail_dataset(df: pd.DataFrame, split: str, shuffle: bool = True) -> tf.data.Dataset:
    """Builds a tf.data.Dataset for the core pass/fail classification task.

    Args:
        df: The DataFrame returned by build_image_level_labels().
        split: Which split to build the dataset for ("train", "valid", "test").
        shuffle: Whether to shuffle the dataset. Should be True for training
            and False for validation/test, to keep evaluation order stable
            and reproducible.

    Returns:
        A batched, prefetched tf.data.Dataset yielding (image, label) pairs,
        where image is a (224, 224, 3) float32 tensor normalised to [0, 1],
        and label is an integer: 0 for pass, 1 for fail.
    """
    split_df = df[df["split"] == split].copy()
    split_df["label"] = (split_df["pass_fail"] == "fail").astype(int)

    paths = split_df["image_path"].tolist()
    labels = split_df["label"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load_and_preprocess(path, label):
        """Reads an image file off disk and prepares it for the model."""
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = image / 255.0  # normalise pixel values to [0, 1]
        return image, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
