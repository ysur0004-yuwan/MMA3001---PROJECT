"""Unit tests for model_utils.py.

Three groups of tests:

1. Class-weight maths, checked against hand-calculated values.
2. Model architecture, checked against figures quoted in the project report
   (e.g. the 2,259,265 vs 27,809 parameter comparison in Section 4), so the
   report and the code cannot silently drift apart.
3. Evaluation functions, checked with stub models that return known
   predictions, so the expected confusion matrix can be worked out by hand.
   This includes the threshold behaviour behind the 0.3 operating point
   chosen in Sections 5-6, and the zero-support (wrinkle) edge case.

Backbones are built with weights=None so the tests run offline; weights do
not change layer shapes or parameter counts.
"""

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

import model_utils as mu


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _labels_df(pass_n, fail_n, valid_pass=5):
    """Builds a minimal label DataFrame with known pass/fail counts."""
    rows = [{"split": "train", "pass_fail": "pass"}] * pass_n
    rows += [{"split": "train", "pass_fail": "fail"}] * fail_n
    rows += [{"split": "valid", "pass_fail": "pass"}] * valid_pass  # must be ignored
    return pd.DataFrame(rows)


def _dataset(images, labels, batch=4):
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch)


class _StubBinaryModel:
    """Duck-typed model returning a fixed fail-probability for every image."""
    def __init__(self, prob):
        self.prob = prob

    def predict(self, images, verbose=0):
        return np.full((len(images), 1), self.prob, dtype="float32")


class _StubMulticlassModel:
    """Duck-typed model that always predicts one fixed class index."""
    def __init__(self, predicted_class, n_classes=5):
        self.predicted_class = predicted_class
        self.n_classes = n_classes

    def predict(self, images, verbose=0):
        probs = np.zeros((len(images), self.n_classes), dtype="float32")
        probs[:, self.predicted_class] = 1.0
        return probs


# --------------------------------------------------------------------------
# class weights
# --------------------------------------------------------------------------
def test_binary_class_weights_match_hand_calculation():
    # 2 pass, 6 fail: total 8 -> pass = 8/(2*2) = 2.0, fail = 8/(2*6) = 0.6667
    w = mu.compute_class_weights(_labels_df(pass_n=2, fail_n=6))
    assert w[0] == pytest.approx(2.0)
    assert w[1] == pytest.approx(8 / 12)


def test_binary_class_weights_ignore_other_splits():
    # The 5 valid-split pass images must not change the training weights.
    w_a = mu.compute_class_weights(_labels_df(2, 6, valid_pass=0))
    w_b = mu.compute_class_weights(_labels_df(2, 6, valid_pass=500))
    assert w_a == w_b


def test_binary_class_weights_upweight_minority_class():
    w = mu.compute_class_weights(_labels_df(pass_n=1, fail_n=9))
    assert w[0] > w[1]


def test_multiclass_class_weights_match_hand_calculation(class_names):
    # counts: loose=1, packaging=6, twisted=1, unsealed=4, wrinkle=0 -> total 12
    labels = (["loose-meat"] + ["packaging-error"] * 6 + ["twisted-meat"]
              + ["unsealed"] * 4)
    df = pd.DataFrame({"split": "train", "extension_label": labels})
    w = mu.compute_multiclass_class_weights(df, class_names)
    assert w[0] == pytest.approx(12 / (5 * 1))
    assert w[1] == pytest.approx(12 / (5 * 6))
    assert w[3] == pytest.approx(12 / (5 * 4))


def test_multiclass_weight_for_absent_class_is_zero_not_an_error(class_names):
    df = pd.DataFrame({"split": "train", "extension_label": ["unsealed", "packaging-error"]})
    w = mu.compute_multiclass_class_weights(df, class_names)
    assert w[4] == 0.0  # wrinkle never appears: no ZeroDivisionError


def test_multiclass_weights_exclude_unlabelled_rows(class_names):
    df = pd.DataFrame({
        "split": "train",
        "extension_label": ["unsealed", "unsealed", None, None],  # pass/multi-defect rows
    })
    w = mu.compute_multiclass_class_weights(df, class_names)
    assert w[3] == pytest.approx(2 / (5 * 2))


# --------------------------------------------------------------------------
# architectures
# --------------------------------------------------------------------------
def test_baseline_model_matches_reported_parameter_count():
    model = mu.build_baseline_model(weights=None)
    assert model.count_params() == 2_259_265  # figure quoted in report Section 4


def test_baseline_backbone_is_frozen_only_head_trains():
    model = mu.build_baseline_model(weights=None)
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    assert trainable == 1280 * 1 + 1  # single Dense(1) head on 1280 features


def test_baseline_output_is_a_probability_per_image():
    model = mu.build_baseline_model(weights=None)
    out = model.predict(np.random.rand(3, 224, 224, 3).astype("float32"), verbose=0)
    assert out.shape == (3, 1)
    assert ((out >= 0) & (out <= 1)).all()


def test_simple_cnn_matches_reported_parameter_count():
    model = mu.build_simple_cnn_model()
    assert model.count_params() == 27_809  # figure quoted in report Section 4


def test_simple_cnn_is_fully_trainable_unlike_the_baseline():
    model = mu.build_simple_cnn_model()
    assert len(model.non_trainable_weights) == 0


def test_defect_type_model_outputs_a_probability_distribution():
    model = mu.build_defect_type_model(num_classes=5, weights=None)
    out = model.predict(np.random.rand(2, 224, 224, 3).astype("float32"), verbose=0)
    assert out.shape == (2, 5)
    assert out.sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-4)  # softmax


def test_defect_type_model_only_head_trains():
    model = mu.build_defect_type_model(num_classes=5, weights=None)
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    assert trainable == 1280 * 5 + 5


# --------------------------------------------------------------------------
# evaluate_pass_fail_model
# --------------------------------------------------------------------------
IMAGES = np.zeros((4, 8, 8, 3), dtype="float32")
LABELS = np.array([0, 0, 1, 1])  # 2 pass, 2 fail


def test_pass_fail_eval_confusion_matrix_for_always_fail_model(capsys):
    res = mu.evaluate_pass_fail_model(_StubBinaryModel(0.9), _dataset(IMAGES, LABELS))
    # Always predicting fail: both real passes become false alarms.
    assert res["confusion_matrix"].tolist() == [[0, 2], [0, 2]]
    assert "Classification report" in capsys.readouterr().out


def test_pass_fail_eval_perfect_model():
    probs = {0: 0.1, 1: 0.9}

    class Perfect:
        def predict(self, images, verbose=0):
            # Recover the label from a marker we stored in pixel (0,0,0).
            return np.array([[probs[int(img[0, 0, 0])]] for img in images], dtype="float32")

    images = IMAGES.copy()
    images[:, 0, 0, 0] = LABELS  # smuggle the true label into the image
    res = mu.evaluate_pass_fail_model(Perfect(), _dataset(images, LABELS))
    assert res["confusion_matrix"].tolist() == [[2, 0], [0, 2]]


def test_threshold_controls_how_readily_images_are_flagged_fail():
    """The mechanism behind the report's 0.3 operating point (Sections 5-6).

    A model that outputs 0.4 'fail' probability misses every defect at the
    default 0.5 threshold but flags every one at 0.3.
    """
    ds = _dataset(IMAGES, LABELS)
    at_default = mu.evaluate_pass_fail_model(_StubBinaryModel(0.4), ds, threshold=0.5)
    at_tuned = mu.evaluate_pass_fail_model(_StubBinaryModel(0.4), ds, threshold=0.3)
    assert at_default["y_pred"].tolist() == [0, 0, 0, 0]  # real defects missed
    assert at_tuned["y_pred"].tolist() == [1, 1, 1, 1]    # real defects caught


def test_pass_fail_eval_returns_aligned_arrays():
    res = mu.evaluate_pass_fail_model(_StubBinaryModel(0.7), _dataset(IMAGES, LABELS, batch=3))
    assert len(res["y_true"]) == len(res["y_pred"]) == len(res["y_pred_probs"]) == 4
    assert res["y_true"].tolist() == LABELS.tolist()


# --------------------------------------------------------------------------
# evaluate_defect_type_model
# --------------------------------------------------------------------------
def test_defect_eval_keeps_zero_support_class_in_report(class_names, capsys):
    """wrinkle has no test examples; it must still appear, not crash or vanish."""
    labels = np.array([0, 1, 1, 3])  # no class 4 (wrinkle) present
    res = mu.evaluate_defect_type_model(
        _StubMulticlassModel(predicted_class=3), _dataset(IMAGES, labels), class_names)
    out = capsys.readouterr().out
    assert res["confusion_matrix"].shape == (5, 5)
    assert "wrinkle" in out
    assert res["confusion_matrix"][:, 4].sum() == 0  # never predicted either


def test_defect_eval_confusion_matrix_values(class_names):
    labels = np.array([0, 1, 3, 3])
    res = mu.evaluate_defect_type_model(
        _StubMulticlassModel(predicted_class=3), _dataset(IMAGES, labels), class_names)
    cm = res["confusion_matrix"]
    assert cm[3, 3] == 2           # both real unsealed correctly found
    assert cm[0, 3] == 1 and cm[1, 3] == 1  # the other two mislabelled as unsealed
    assert cm.sum() == 4


# --------------------------------------------------------------------------
# train_baseline_model (smoke test on a tiny stand-in model)
# --------------------------------------------------------------------------
def test_training_runs_records_history_and_saves_checkpoint(tmp_path):
    tiny = tf.keras.Sequential([
        tf.keras.Input(shape=(8, 8, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    tiny.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    rng = np.random.default_rng(0)
    x = rng.random((16, 8, 8, 3), dtype="float32")
    y = rng.integers(0, 2, 16)
    ckpt = tmp_path / "best.keras"

    history = mu.train_baseline_model(
        tiny, _dataset(x, y, 8), _dataset(x, y, 8),
        class_weight={0: 1.0, 1: 1.0}, epochs=2, checkpoint_path=str(ckpt))

    assert "loss" in history.history and "val_loss" in history.history
    assert len(history.history["loss"]) >= 1
    assert ckpt.exists()  # checkpoint written, protecting against Colab disconnects
