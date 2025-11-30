import tensorflow as tf
from tensorflow import keras

def train_mnist(
    model: keras.Model,
    train_ds,
    val_ds,
    epochs: int = 5,
    lr: float = 1e-3,
    log_dir: str | None = None,
):
    callbacks = []

    if log_dir is not None:
        tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir)
        callbacks.append(tensorboard_cb)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )
    return history
