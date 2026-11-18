"""Stage 1: binary CNN. Input (128,128,3) raw 0-255 pixels; Rescaling(1/255) is inside the model.
Output: one sigmoid unit = P(Non_Arecanut) (class index 1 of VERIFICATION_CLASS_NAMES)."""
import tensorflow as tf
from tensorflow.keras import layers, models

import config


def build_verification_model(input_shape=(*config.VERIFICATION_IMAGE_SIZE, 3), learning_rate=1e-3):
    inp = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inp)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="verification_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model
