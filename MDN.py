import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import LSTM, RepeatVector, BatchNormalization, Conv2D, TimeDistributed, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1, l2

from tensorflow_probability import distributions as tfd

from tensorflow.keras.layers import Dense, Activation, Concatenate

import numpy as np

import utils

# MOST OF THIS CODE originally belongs to https://github.com/oborchers/Medium_Repo

def nnelu(input):
    """ Non-Negative Exponential Linear Unit 
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def slice_parameter_vectors(parameter_vector, components=3, no_parameters=3):
    """ Returns an unpacked list of paramter vectors 
    """
    return [parameter_vector[:, i * components:(i + 1) * components] for i in range(no_parameters)]


def gnll_loss(y, parameter_vector):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    alpha, mu, sigma = slice_parameter_vectors(parameter_vector, 3, 3)  # Unpack parameter vectors

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma))

    log_likelihood = gm.log_prob(tf.transpose(y))  # Evaluate log-probability of y

    return -tf.reduce_mean(log_likelihood, axis=-1)


tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})


def mdn_eval_corr(y, y_pred, axis=-1):
    alpha_pred, mu_pred, sigma_pred = slice_parameter_vectors(y_pred, 3, 3)
    x = tf.reduce_sum(alpha_pred * mu_pred, axis=-1)

    x = tf.reshape(x, (-1,))
    y = tf.reshape(y, (-1,))

    n = tf.cast(tf.shape(x)[axis], x.dtype)
    xsum = tf.reduce_sum(x, axis=axis)
    ysum = tf.reduce_sum(y, axis=axis)
    xmean = xsum / n
    ymean = ysum / n

    xsqsum = tf.reduce_sum(tf.math.squared_difference(x, xmean), axis=axis)
    ysqsum = tf.reduce_sum(tf.math.squared_difference(y, ymean), axis=axis)
    cov = tf.reduce_sum((x - xmean) * (y - ymean), axis=axis)
    corr = cov / (tf.math.sqrt(xsqsum * ysqsum) + 1e-15)
    return tf.convert_to_tensor(tf.reduce_mean(corr), dtype=tf.float32)


@tf.function
def mdn_eval_corr_map(y, y_pred, axis=-1):
    alpha_pred, mu_pred, sigma_pred = slice_parameter_vectors(y_pred, 3, 3)

    amax = tf.argmax(alpha_pred, axis=1)
    b = tf.range(len(y), dtype=tf.int64)
    d = tf.concat([b[..., None], amax[..., None]], axis=1)
    # x = tf.concat([mu_pred[None, i, k] for i, k in enumerate()], axis=0)
    x = tf.gather_nd(mu_pred, d)

    x = tf.reshape(x, (-1,))
    y = tf.reshape(y, (-1,))

    n = tf.cast(tf.shape(x)[axis], x.dtype)
    xsum = tf.reduce_sum(x, axis=axis)
    ysum = tf.reduce_sum(y, axis=axis)
    xmean = xsum / n
    ymean = ysum / n

    xsqsum = tf.reduce_sum(tf.math.squared_difference(x, xmean), axis=axis)
    ysqsum = tf.reduce_sum(tf.math.squared_difference(y, ymean), axis=axis)
    cov = tf.reduce_sum((x - xmean) * (y - ymean), axis=axis)
    corr = cov / (tf.math.sqrt(xsqsum * ysqsum) + 1e-15)
    return tf.convert_to_tensor(tf.reduce_mean(corr), dtype=tf.float32)


def gnll_eval(y, alpha, mu, sigma):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma))
    log_likelihood = gm.log_prob(tf.transpose(y))
    return -tf.reduce_mean(log_likelihood, axis=-1)


def eval_mdn_model(y_pred, y_test):
    alpha_pred, mu_pred, sigma_pred = slice_parameter_vectors(y_pred, 3, 3)
    _mse = tf.losses.mean_squared_error(np.multiply(alpha_pred, mu_pred).sum(axis=-1)[:, np.newaxis], y_test).numpy().mean()
    _corr_mle = utils.my_corrcoef(np.multiply(alpha_pred, mu_pred).sum(axis=-1).squeeze(), y_test.squeeze())

    y_pred_map = np.array([mu_pred[i, k] for i, k in enumerate(np.argmax(alpha_pred, axis=1))])
    _corr_map = utils.my_corrcoef(y_pred_map.squeeze(), y_test.squeeze())
    _nll = gnll_eval(y_test.astype(np.float32), alpha_pred, mu_pred, sigma_pred).numpy()[0]

    return _mse, _nll, _corr_mle, _corr_map
