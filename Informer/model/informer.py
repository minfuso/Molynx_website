import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

from Informer.layers.positional_encoding import SinusoidalPE
from Informer.layers.encoder import InformerEncoder
from Informer.layers.prediction import PredictionHead

@register_keras_serializable(package="Informer")
class Informer(tf.keras.Model):
    def __init__(self, d_model, num_heads, d_ff, num_layers, horizons, dropout=0.1, u=20, activation_last_layer="sigmoid", **kwargs):
        super().__init__(**kwargs)
        # stocker les hyperparams pour get_config()
        self.d_model = d_model                 # <--- important
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.horizons = horizons
        self.dropout = dropout
        self.u = u
        self.activation_last_layer = activation_last_layer
        
        self.input_proj = tf.keras.layers.Dense(d_model)  # projette tes features -> d_model
        self.pe = SinusoidalPE(d_model)  # positional encoding
        self.encoder = InformerEncoder(d_model, num_heads, d_ff, num_layers, dropout, u)
        self.head = PredictionHead(d_model, horizons, dropout, activation_last_layer=activation_last_layer)

    def call(self, x, training=None):
        # x: (B, T, d_model) déjà normalisé
        x = self.input_proj(x)  # projette les features -> d_model (B, T, d_model)
        x = self.pe(x)                       # ajoute l'encodage positionnel
        x = self.encoder(x, training=training)  # encodeur Informer
        x = self.head(x, training=training)     # prédicteur multi-horizon
        return x  # (B, horizons)
    
    def get_config(self):
        base = super().get_config()
        base.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "num_layers": self.num_layers,
            "horizons": self.horizons,
            "dropout": self.dropout,
            "u": self.u,
            "activation_last_layer": self.activation_last_layer
        })
        return base
    
    @classmethod
    def from_config(cls, config):
        # Keras peut injecter des clés génériques (name, trainable, dtype, dynamic, build_config, etc.)
        # On ne passe au constructeur que les hyperparams que __init__ attend.
        known_keys = {
            "d_model", "num_heads", "d_ff", "num_layers", "horizons",
            "dropout", "u", "activation_last_layer"
        }
        ctor_kwargs = {k: v for k, v in config.items() if k in known_keys}

        # NB: on ignore volontairement les autres clés (name/trainable/dtype...), car
        # super().__init__(**kwargs) les prendra si et seulement si tu ajoutes **kwargs à __init__
        return cls(**ctor_kwargs)
