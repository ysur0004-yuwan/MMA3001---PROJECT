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

import pandas as pd
import tensorflow as tf

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
