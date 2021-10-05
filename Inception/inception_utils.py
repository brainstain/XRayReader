import tensorflow as tf
from keras.backend.common import epsilon
from keras import backend as K


def weighted_binary_crossentropy(target, output, pos_weight, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.
    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=pos_weight)


# todo make these configurable for batch size
batch_size = 50
def correct_positive(y_true, y_pred):
    equal = K.equal(y_true, K.round(y_pred))
    correct_positives = tf.where(equal, y_true, tf.zeros([batch_size, 15]))
    return tf.divide(tf.reduce_sum(correct_positives, 1), tf.reduce_sum(y_true, 1))


def false_positive(y_true, y_pred):
    rounded = tf.round(y_pred)
    equal = tf.not_equal(y_true, rounded)
    false_positives = tf.where(equal, rounded, tf.zeros([batch_size, 15]))
    return tf.reduce_sum(false_positives, 1)


def false_negative(y_true, y_pred):
    rounded = tf.round(y_pred)
    equal = tf.not_equal(y_true, rounded)
    false_negatives = tf.where(equal, y_true, tf.zeros([batch_size, 15]))
    return tf.reduce_sum(false_negatives, 1)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)