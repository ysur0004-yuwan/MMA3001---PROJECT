"""model_utils.py

Model-building and training utilities for the pass/fail classification task.

The baseline model uses transfer learning with a frozen MobileNetV2 backbone
(pretrained on ImageNet) plus a small trainable classification head. This is
a deliberate "simple first" choice: freezing the backbone means we are only
training a handful of new parameters, which trains fast and is unlikely to
overfit on a dataset of this size — a sensible baseline before attempting
anything more complex (see Section 4, Alternative Solutions, for the planned
comparison against this baseline).

Project: MMA3001 Individual Project — Automated Classification of Pork
Rasher Packaging Errors.

Docstring style: Google style, consistent with data_utils.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Must match IMAGE_SIZE in data_utils.py, since the model's input layer has
# to agree with the shape produced by the data pipeline.
INPUT_SHAPE = (224, 224, 3)


def build_baseline_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """Builds the baseline pass/fail classifier.

    Architecture: a frozen MobileNetV2 backbone (ImageNet weights) followed
    by global average pooling, dropout for regularisation, and a single
    sigmoid output unit for binary classification (0 = pass, 1 = fail).

    Args:
        input_shape: Shape of the input images, e.g. (224, 224, 3). Must
            match the image size produced by data_utils.make_pass_fail_dataset.
        learning_rate: Learning rate for the Adam optimiser. 1e-4 is a
            conservative default appropriate for fine-tuning a small head on
            top of a frozen pretrained backbone.

    Returns:
        A compiled tf.keras.Model, ready to call .fit() on.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    # Baseline choice: freeze the backbone entirely. We are only training the
    # new classification head, which keeps the baseline fast and simple.
    # Fine-tuning (unfreezing some backbone layers) is left for the
    # "improved" comparison model in Section 4/6, not the baseline.
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # light regularisation against overfitting
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="baseline_pass_fail_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def compute_class_weights(df: pd.DataFrame, split: str = "train") -> dict[int, float]:
    """Computes inverse-frequency class weights for the pass/fail task.

    The dataset is imbalanced (~80% fail, ~20% pass across all splits, per
    the Section 2 data audit). Without correction, a model could achieve
    high accuracy by simply always predicting "fail" without learning
    anything useful. Class weighting penalises mistakes on the minority
    class (pass) more heavily during training, to counteract this.

    Args:
        df: The DataFrame returned by data_utils.build_image_level_labels().
        split: Which split to compute weights from. Should always be "train"
            — weights are a training-time correction and must not be
            computed from validation/test data.

    Returns:
        A dict mapping the integer label (0 = pass, 1 = fail) to its class
        weight, suitable for passing directly to model.fit(class_weight=...).
    """
    split_df = df[df["split"] == split]
    counts = split_df["pass_fail"].value_counts()
    total = counts.sum()
    n_classes = 2

    # Standard inverse-frequency formula: weight = total / (n_classes * class_count)
    weight_pass = total / (n_classes * counts.get("pass", 1))
    weight_fail = total / (n_classes * counts.get("fail", 1))

    return {0: weight_pass, 1: weight_fail}  # 0 = pass, 1 = fail (matches make_pass_fail_dataset)


def train_baseline_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weight: dict[int, float] | None = None,
    epochs: int = 15,
    checkpoint_path: str = "baseline_model.keras",
) -> tf.keras.callbacks.History:
    """Trains the baseline model with early stopping and checkpointing.

    Args:
        model: A compiled model, e.g. from build_baseline_model().
        train_ds: Training dataset, e.g. from
            data_utils.make_pass_fail_dataset(df, split="train", shuffle=True).
        val_ds: Validation dataset (shuffle=False).
        class_weight: Optional class weights, e.g. from compute_class_weights().
        epochs: Maximum number of training epochs. Early stopping will likely
            halt training before this is reached.
        checkpoint_path: Where to save the best model (by validation loss)
            during training.

    Returns:
        The Keras History object, containing per-epoch training and
        validation metrics — used directly for the Validation section's
        learning curves and final-epoch metrics.
    """
    callbacks = [
        # Stop training once validation loss stops improving, and restore
        # the best-performing weights rather than the final epoch's weights.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        # Save a checkpoint of the best model seen so far, in case the
        # session disconnects mid-training (a real risk on free Colab).
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    return history


def evaluate_pass_fail_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: tuple[str, str] = ("pass", "fail"),
    threshold: float = 0.5,
) -> dict:
    """Evaluates a trained pass/fail model on a held-out dataset.

    This is the function used for final reporting (Section 5, Validation),
    and should always be called on the TEST split, never on validation or
    training data, to give an honest estimate of real-world performance.
    Validation data was already used during training to pick the best
    checkpoint (see train_baseline_model), so it cannot also serve as an
    unbiased final result.

    Args:
        model: A trained model, e.g. the output of train_baseline_model().
        dataset: The dataset to evaluate on — should be the test split,
            built with shuffle=False so predictions stay aligned with the
            correct images (order doesn't actually matter for metrics, but
            shuffle=False is good practice for reproducible evaluation).
        class_names: Human-readable names for (negative_class, positive_class),
            i.e. (pass, fail), used to label the classification report.
        threshold: Probability threshold above which a prediction counts as
            "fail" (the positive class). 0.5 is the standard default; a
            different threshold could be justified in Section 6
            (Optimisation) if precision/recall trade-offs are explored.

    Returns:
        A dict with keys: "y_true", "y_pred", "y_pred_probs" (numpy arrays),
        and "confusion_matrix" (2x2 numpy array), for further analysis or
        plotting if needed.
    """
    y_true = []
    y_pred_probs = []

    # Iterate the dataset once, collecting true labels and predicted
    # probabilities batch by batch.
    for images, labels in dataset:
        y_true.extend(labels.numpy())
        batch_probs = model.predict(images, verbose=0)
        y_pred_probs.extend(batch_probs.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= threshold).astype(int)

    # zero_division=0 avoids a crash if a class is ever entirely absent or
    # entirely unpredicted — important given how thin some classes are
    # elsewhere in this project (see Section 2, wrinkle class).
    report = classification_report(
        y_true, y_pred, target_names=list(class_names), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print("Classification report (test set):")
    print(report)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(f"               pred_{class_names[0]}   pred_{class_names[1]}")
    print(f"true_{class_names[0]:<8}  {cm[0][0]:>10}  {cm[0][1]:>10}")
    print(f"true_{class_names[1]:<8}  {cm[1][0]:>10}  {cm[1][1]:>10}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_probs": y_pred_probs,
        "confusion_matrix": cm,
    }
