import tensorflow as tf
from tensorflow.keras import layers
from keras.saving import register_keras_serializable

@register_keras_serializable(package="SinusoidalPE")
class SinusoidalPE(layers.Layer):
    """Encodage positionnel sin/cos (Vaswani et al., 2017)."""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def call(self, x):
        # x: (B, T, d_model) — B=batch, T=longueur séquence, d_model=dim embedding
        T = tf.shape(x)[1]
        d = self.d_model

        # positions: [0, 1, ..., T-1]  shape (T, 1)
        pos = tf.cast(tf.range(T)[:, None], tf.float32)

        # indices de dimensions: [0, 1, ..., d-1] shape (1, d)
        i = tf.cast(tf.range(d)[None, :], tf.float32)

        # angle_rates = 1 / 10000^{2i/d}
        angle_rates = tf.pow(10000.0, - (tf.floor(i / 2.0) * 2.0) / tf.cast(d, tf.float32))
        angle_rads = pos * angle_rates  # (T, d)

        # Appliquer sin sur dimensions paires (2k), cos sur impaires (2k+1)
        sines = tf.sin(angle_rads[:, 0::2])
        coses = tf.cos(angle_rads[:, 1::2])

        # Recomposer en (T, d): intercaler sin/cos
        pe = tf.concat(
            [tf.reshape(tf.stack([sines, coses], axis=-1), (T, -1)),
             tf.zeros((T, d - tf.shape(tf.reshape(tf.stack([sines, coses], axis=-1), (T, -1)))[1]))],
            axis=-1
        )
        pe = pe[:, :d]  # sécurité si d est impair

        return x + pe[None, ...]  # broadcast sur le batch
    
    def get_config(self):
        base = super().get_config()
        base.update({
            "d_model": self.d_model
        })
        return base
    
    @classmethod
    def from_config(cls, config):
        # on filtre juste ce que __init__ accepte
        known_keys = {"d_model"}
        ctor_kwargs = {k: v for k, v in config.items() if k in known_keys}
        return cls(**ctor_kwargs)
