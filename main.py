"""
TensorFlow & Keras — Standalone Demo
Covers: Tensors, Sequential API, Functional API, tf.data, training, evaluation,
        overfitting mitigation, and model saving/loading.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress verbose TF logs

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────
# 1. Tensors & Variables
# ─────────────────────────────────────────────
def demo_tensors():
    print("\n=== 1. Tensors & Variables ===")

    scalar = tf.constant(3.14)
    vector = tf.constant([1, 2, 3])
    matrix = tf.constant([[1, 2], [3, 4]])
    tensor3d = tf.zeros((2, 3, 4))

    print(f"Scalar: {scalar.numpy():.2f}")
    print(f"Vector shape: {vector.shape}")
    print(f"Matrix:\n{matrix.numpy()}")
    print(f"3D tensor shape: {tensor3d.shape}")

    # Arithmetic
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant([4.0, 5.0, 6.0])
    print(f"Element-wise multiply: {(a * b).numpy()}")
    print(f"Dot product: {tf.reduce_sum(a * b).numpy():.1f}")

    # Variables (mutable)
    var = tf.Variable([1.0, 2.0, 3.0])
    var.assign_add([0.5, 0.5, 0.5])
    print(f"Variable after assign_add: {var.numpy()}")


# ─────────────────────────────────────────────
# 2. tf.data Pipeline
# ─────────────────────────────────────────────
def demo_tf_data():
    print("\n=== 2. tf.data Pipeline ===")

    data = np.arange(20, dtype=np.float32)
    labels = (data > 10).astype(np.float32)

    dataset = (
        tf.data.Dataset.from_tensor_slices((data, labels))
        .shuffle(buffer_size=20, seed=42)
        .batch(4)
        .prefetch(tf.data.AUTOTUNE)
    )

    for batch_x, batch_y in dataset.take(2):
        print(f"  Batch X: {batch_x.numpy()}, Y: {batch_y.numpy()}")


# ─────────────────────────────────────────────
# 3. Sequential API — Binary Classification
# ─────────────────────────────────────────────
def demo_sequential():
    print("\n=== 3. Sequential API — Binary Classification ===")

    # Synthetic dataset
    np.random.seed(0)
    X = np.random.randn(500, 8).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

    split = 400
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = keras.Sequential([
        layers.Dense(32, activation="relu", input_shape=(8,)),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
    return model, X_val, y_val


# ─────────────────────────────────────────────
# 4. Functional API — Multi-output Model
# ─────────────────────────────────────────────
def demo_functional():
    print("\n=== 4. Functional API — Multi-output Model ===")

    inputs = keras.Input(shape=(10,), name="input")
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)

    output_cls = layers.Dense(1, activation="sigmoid", name="classifier")(x)
    output_reg = layers.Dense(1, name="regressor")(x)

    model = keras.Model(inputs=inputs, outputs=[output_cls, output_reg])

    model.compile(
        optimizer="adam",
        loss={"classifier": "binary_crossentropy", "regressor": "mse"},
        metrics={"classifier": "accuracy"},
    )

    print(f"Inputs:  {[i.shape for i in model.inputs]}")
    print(f"Outputs: {[o.shape for o in model.outputs]}")


# ─────────────────────────────────────────────
# 5. Overfitting Mitigation
# ─────────────────────────────────────────────
def demo_regularization():
    print("\n=== 5. Overfitting Mitigation ===")

    model = keras.Sequential([
        layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-3),
            input_shape=(8,),
        ),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("Regularized model built with L2 + Dropout + BatchNorm")
    print(f"Total params: {model.count_params():,}")


# ─────────────────────────────────────────────
# 6. Save & Load
# ─────────────────────────────────────────────
def demo_save_load(model, X_val, y_val):
    print("\n=== 6. Save & Load ===")

    save_path = "/tmp/demo_model.keras"
    model.save(save_path)
    print(f"Model saved to {save_path}")

    loaded = keras.models.load_model(save_path)
    _, acc_original = model.evaluate(X_val, y_val, verbose=0)
    _, acc_loaded = loaded.evaluate(X_val, y_val, verbose=0)
    print(f"Original accuracy: {acc_original:.4f}")
    print(f"Loaded   accuracy: {acc_loaded:.4f}")
    assert abs(acc_original - acc_loaded) < 1e-5, "Accuracy mismatch after load!"
    print("Save/load verified successfully.")


# ─────────────────────────────────────────────
# 7. Optimizers & Learning Rate Schedule
# ─────────────────────────────────────────────
def demo_optimizers():
    print("\n=== 7. Optimizers & LR Scheduling ===")

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=1000,
        decay_rate=0.9,
    )

    optimizers = {
        "Adam (lr=1e-3)":   keras.optimizers.Adam(learning_rate=1e-3),
        "SGD (momentum)":   keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        "RMSprop":          keras.optimizers.RMSprop(learning_rate=1e-3),
        "Adam (LR sched.)": keras.optimizers.Adam(learning_rate=lr_schedule),
    }

    for name, opt in optimizers.items():
        print(f"  {name}: {opt.__class__.__name__}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)

    demo_tensors()
    demo_tf_data()
    model, X_val, y_val = demo_sequential()
    demo_functional()
    demo_regularization()
    demo_save_load(model, X_val, y_val)
    demo_optimizers()

    print("\nAll demos completed successfully.")
