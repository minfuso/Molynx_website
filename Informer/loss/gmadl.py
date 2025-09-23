import tensorflow as tf

import tensorflow as tf

class GMADL(tf.keras.losses.Loss):
    def __init__(self, a=1.0, b=1.0, name="GMADL"):
        super().__init__(name=name)
        self.a = a  # contrôle la pente de la tanh
        self.b = b  # pondération par magnitude

    def call(self, y_true, y_pred):
        """
        y_true : rendements réels (float32, peut être <0 ou >0)
        y_pred : rendements prédits (float32, sortie linéaire du modèle)
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Terme directionnel différentiable
        directional_term = 1.0 - tf.tanh(self.a * y_true * y_pred)

        # Pondération par magnitude
        weight = 1.0 + self.b * tf.abs(y_true)

        return tf.reduce_mean(directional_term * weight)



class GMADLBinary(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=1.0, name="GMADLBinary"):
        super().__init__(name=name)
        self.alpha = alpha  # contrôle la pente de la fonction tanh
        self.beta = beta    # pondération supplémentaire

    def call(self, y_true, y_pred):
        """
        y_true : (batch,) valeurs {0,1}
        y_pred : (batch,) probas sigmoïdes ∈ [0,1]
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Encode la direction : 0 -> -1, 1 -> +1
        sign_true = 2.0 * y_true - 1.0

        # Transforme les probas en direction "douce" via tanh
        sign_pred = tf.tanh(self.alpha * (y_pred - 0.5) * 2.0)

        # Erreur directionnelle
        directional_error = tf.abs(sign_true - sign_pred)

        # Pondération (ici, simple, car pas de magnitude des rendements dispo)
        weighted_error = (1.0 + self.beta) * directional_error

        return tf.reduce_mean(weighted_error)
