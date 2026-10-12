"""Unit tests for data_utils.py.

These tests verify the label-conversion logic that every downstream result
depends on: if image-level labels are derived incorrectly, every metric in
the project report would be wrong regardless of model quality. Each test
targets a specific documented behaviour from Section 2 of the project
documentation (label derivation policy and handling of invalid inputs).
"""

import numpy as np
import pandas as pd
import pytest

import data_utils as du


# --------------------------------------------------------------------------
# load_class_names
# --------------------------------------------------------------------------
def test_load_class_names_returns_names_in_id_order(fake_dataset, class_names):
    assert du.load_class_names(fake_dataset) == class_names


def test_load_class_names_raises_when_yaml_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        du.load_class_names(tmp_path)


# --------------------------------------------------------------------------
# parse_yolo_label_file
# --------------------------------------------------------------------------
def test_parse_label_file_returns_one_id_per_box(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("3 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n3 0.7 0.7 0.1 0.1\n")
    assert du.parse_yolo_label_file(f) == [3, 1, 3]


def test_parse_label_file_empty_file_means_no_defects(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    assert du.parse_yolo_label_file(f) == []


def test_parse_label_file_missing_file_treated_as_no_defects(tmp_path):
    assert du.parse_yolo_label_file(tmp_path / "does_not_exist.txt") == []


def test_parse_label_file_skips_malformed_lines_without_crashing(tmp_path, caplog):
    f = tmp_path / "bad.txt"
    f.write_text("3 0.5 0.5 0.1 0.1\nnot_a_number 0.1 0.1\n2 0.4 0.4 0.1 0.1\n")
    with caplog.at_level("WARNING"):
        result = du.parse_yolo_label_file(f)
    assert result == [3, 2]                      # valid lines kept
    assert "Malformed annotation line" in caplog.text  # and the problem was logged


# --------------------------------------------------------------------------
# is_image_readable
# --------------------------------------------------------------------------
def test_is_image_readable_true_for_valid_jpeg(fake_dataset):
    assert du.is_image_readable(fake_dataset / "train/images/pass_a.jpg") is True


def test_is_image_readable_false_for_corrupt_file(fake_dataset):
    assert du.is_image_readable(fake_dataset / "train/images/corrupt.jpg") is False


# --------------------------------------------------------------------------
# build_image_level_labels  (the core label-derivation policy)
# --------------------------------------------------------------------------
@pytest.fixture
def labels_df(fake_dataset, class_names):
    return du.build_image_level_labels(fake_dataset, class_names)


def _row(df, filename):
    return df[df["image_path"].str.endswith(filename)].iloc[0]


def test_labels_dataframe_has_expected_columns(labels_df):
    assert list(labels_df.columns) == [
        "image_path", "split", "pass_fail", "is_multi_defect", "extension_label"
    ]


def test_corrupt_image_is_excluded_entirely(labels_df):
    assert not labels_df["image_path"].str.endswith("corrupt.jpg").any()
    assert len(labels_df) == 8  # 9 files created, 1 corrupt


def test_images_per_split_match_fixture(labels_df):
    counts = labels_df["split"].value_counts().to_dict()
    assert counts == {"train": 5, "valid": 2, "test": 1}


def test_empty_label_file_gives_pass(labels_df):
    row = _row(labels_df, "pass_a.jpg")
    assert row["pass_fail"] == "pass"
    # pandas stores a missing string label as NaN, so test with isna, not "is None"
    assert pd.isna(row["extension_label"])
    assert not row["is_multi_defect"]


def test_missing_label_file_gives_pass(labels_df):
    assert _row(labels_df, "pass_no_label.jpg")["pass_fail"] == "pass"


def test_single_defect_gets_fail_and_class_name(labels_df):
    row = _row(labels_df, "unsealed_a.jpg")
    assert row["pass_fail"] == "fail"
    assert row["extension_label"] == "unsealed"
    assert not row["is_multi_defect"]


def test_repeated_boxes_of_one_class_are_not_multi_defect(labels_df):
    """Two boxes of the SAME class is one defect type, not a multi-defect image.

    This is why the image-level class counts in Section 2 differ from the raw
    bounding-box counts, and the distinction must not be lost.
    """
    row = _row(labels_df, "pkg_two_boxes.jpg")
    assert row["pass_fail"] == "fail"
    assert row["extension_label"] == "packaging-error"
    assert not row["is_multi_defect"]


def test_multi_defect_image_fails_but_is_excluded_from_extension(labels_df):
    """Documented policy: multi-defect images stay in pass/fail, leave extension."""
    row = _row(labels_df, "multi_a.jpg")
    assert row["pass_fail"] == "fail"
    assert row["is_multi_defect"]
    assert pd.isna(row["extension_label"])


def test_every_fail_image_is_accounted_for(labels_df):
    """fail = single-label fail + multi-defect, mirroring the 2,839 = 2,748 + 91 check."""
    fails = labels_df[labels_df["pass_fail"] == "fail"]
    single = fails["extension_label"].notna().sum()
    multi = fails["is_multi_defect"].sum()
    assert len(fails) == single + multi


def test_missing_split_folder_is_skipped_not_fatal(tmp_path, class_names, caplog):
    import yaml
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": class_names}))
    with caplog.at_level("WARNING"):
        df = du.build_image_level_labels(tmp_path, class_names)
    assert len(df) == 0
    assert "not found" in caplog.text


def test_summarise_labels_prints_key_sections(labels_df, capsys):
    du.summarise_labels(labels_df)
    out = capsys.readouterr().out
    assert "Images per split" in out
    assert "Pass/fail counts" in out
    assert "Multi-defect images excluded" in out


# --------------------------------------------------------------------------
# tf.data pipelines
# --------------------------------------------------------------------------
def _collect(ds):
    xs, ys = [], []
    for x, y in ds:
        xs.append(x.numpy())
        ys.extend(y.numpy().tolist())
    return np.concatenate(xs), ys


def test_pass_fail_dataset_shapes_range_and_labels(labels_df):
    ds = du.make_pass_fail_dataset(labels_df, "train", shuffle=False)
    images, labels = _collect(ds)
    assert images.shape == (5, 224, 224, 3)
    assert images.dtype == np.float32
    assert 0.0 <= images.min() and images.max() <= 1.0
    # train fixture: 2 pass, 3 fail
    assert sorted(labels) == [0, 0, 1, 1, 1]


def test_pass_fail_dataset_unshuffled_order_is_stable(labels_df):
    first = _collect(du.make_pass_fail_dataset(labels_df, "train", shuffle=False))[1]
    second = _collect(du.make_pass_fail_dataset(labels_df, "train", shuffle=False))[1]
    assert first == second


def test_defect_type_dataset_excludes_pass_and_multi_defect(labels_df, class_names):
    ds = du.make_defect_type_dataset(labels_df, "train", class_names, shuffle=False)
    images, labels = _collect(ds)
    # train has exactly two single-label fail images: unsealed_a (3), pkg_two_boxes (1)
    assert images.shape[0] == 2
    assert sorted(labels) == [1, 3]


def test_defect_type_dataset_label_indices_follow_class_name_order(labels_df, class_names):
    ds = du.make_defect_type_dataset(labels_df, "valid", class_names, shuffle=False)
    _, labels = _collect(ds)
    assert labels == [class_names.index("loose-meat")]  # == 0


# --------------------------------------------------------------------------
# shuffle=True branch (the path actually used when training)
# --------------------------------------------------------------------------
def test_shuffled_pass_fail_dataset_keeps_exactly_the_same_examples(labels_df):
    """Shuffling may reorder examples but must never add, drop or alter any."""
    plain = _collect(du.make_pass_fail_dataset(labels_df, "train", shuffle=False))[1]
    shuffled = _collect(du.make_pass_fail_dataset(labels_df, "train", shuffle=True))[1]
    assert sorted(shuffled) == sorted(plain)


def test_shuffled_defect_type_dataset_keeps_exactly_the_same_examples(labels_df, class_names):
    plain = _collect(du.make_defect_type_dataset(labels_df, "train", class_names, shuffle=False))[1]
    shuffled = _collect(du.make_defect_type_dataset(labels_df, "train", class_names, shuffle=True))[1]
    assert sorted(shuffled) == sorted(plain)
