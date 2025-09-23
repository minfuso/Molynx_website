import tensorflow as tf

# RMSE
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# MAPE
def mape(y_true, y_pred):
    epsilon = 1e-8
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100