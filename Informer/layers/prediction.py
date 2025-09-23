import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable(package="PredictionHead")
class PredictionHead(tf.keras.layers.Layer):
    def __init__(self, d_model, horizons, dropout=0.1, activation_last_layer=None, **kwargs):
        super().__init__(**kwargs)
        # stocker les hyperparams pour get_config()
        self.d_model = d_model
        self.horizons = horizons
        self.dropout = dropout
        self.activation_last_layer = activation_last_layer

        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(horizons, activation=activation_last_layer)
        ])

    def call(self, x, training=False):
        # x: (B, T', d_model)
        x = self.pool(x)       # (B, d_model)
        x = self.ffn(x, training=training)  # (B, horizons)
        return x
    
    def get_config(self):
        base = super().get_config()
        base.update({
            "d_model": self.d_model,
            "horizons": self.horizons,
            "dropout": self.dropout,
            "activation_last_layer": self.activation_last_layer
        })
        return base
    
    @classmethod
    def from_config(cls, config):
        # filtrer uniquement les arguments que __init__ accepte
        known_keys = {"d_model", "horizons", "dropout", "activation_last_layer"}
        ctor_kwargs = {k: v for k, v in config.items() if k in known_keys}
        return cls(**ctor_kwargs)
