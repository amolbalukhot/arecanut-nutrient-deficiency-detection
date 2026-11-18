"""Stage 2: MobileNetV2 transfer learning (the only Stage 2 architecture).
Input (224,224,3) already passed through mobilenet_v2.preprocess_input ([-1,1]) OUTSIDE the model.
Output: softmax over config.CLASS_NAMES in that exact order."""
import tensorflow as tf
from tensorflow.keras import layers, models

import config

BASE_NAME = "mobilenetv2_1.00_224"


def build_mobilenetv2(input_shape=(*config.NUTRIENT_IMAGE_SIZE, 3), pretrained=True, learning_rate=1e-3):
    """pretrained=False is the explicit offline option; random init needs much more data."""
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False,
                                             weights="imagenet" if pretrained else None)
    base.trainable = False
    inp = layers.Input(shape=input_shape)
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(config.CLASS_NAMES), activation="softmax")(x)
    model = models.Model(inp, out, name="mobilenetv2_classifier")
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def unfreeze_for_finetuning(model, last_n_layers=30, learning_rate=1e-5):
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    base.trainable = True
    for layer in base.layers[:-last_n_layers]:
        layer.trainable = False
    for layer in base.layers:  
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, learning_rate)
