import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable(package="DistillingLayer")
class DistillingLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

        self.conv = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=3,
            padding="same",
            activation="relu"
        )
        self.pool = tf.keras.layers.MaxPooling1D(pool_size=2)

    def call(self, x):
        # x: (B, T, d_model)
        x = self.conv(x)   # (B, T, d_model)
        x = self.pool(x)   # (B, T/2, d_model)
        return x

    def get_config(self):
        base = super().get_config()
        base.update({
            "d_model": self.d_model
        })
        return base

    @classmethod
    def from_config(cls, config):
        known_keys = {"d_model"}
        ctor_kwargs = {k: v for k, v in config.items() if k in known_keys}
        return cls(**ctor_kwargs)

