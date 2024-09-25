import tensorflow as tf
import math

def beta_pdf(x, alpha, beta, loc=0.0, scale=1.0):

    alpha = tf.convert_to_tensor(alpha, tf.float32)
    beta = tf.convert_to_tensor(beta, tf.float32)
    loc = tf.convert_to_tensor(loc, tf.float32)
    scale = tf.convert_to_tensor(scale, tf.float32)

    x = (x-loc)/scale

    log_unnormalized = (tf.math.xlogy(alpha- 1., x) +
                        tf.math.xlog1py(beta - 1., -x))
    log_normalization = tf.math.lbeta([alpha, beta])
    log_prob = log_unnormalized - log_normalization

    # force zero outside support
    log_prob = tf.where((x >= 0) & (x <= 1), log_prob, -float('inf'))
    
    return tf.divide(tf.exp(log_prob), scale)

def weighted_mean_tf(x, w, sumw):
    return tf.reduce_sum(w * x) / sumw

def weighted_var_tf(x, mean, w, sumw):
    return tf.reduce_sum(w*(x - mean)**2)/sumw

def estimator_beta_tf(data, weights):

    sumw = tf.reduce_sum(weights)

    data = tf.debugging.assert_all_finite(data, "Data in estimator not finite")
    weights = tf.debugging.assert_all_finite(weights, "Weights in estimator not finite")
    
    def mm(data, weights, sumw):

        mean = weighted_mean_tf(data, weights, sumw)
        var = weighted_var_tf(data, mean, weights, sumw)

        mean = tf.debugging.assert_all_finite(mean, "Mean in estimator not finite")
        var = tf.debugging.assert_all_finite(var, "Var in estimator not finite")
        var = tf.maximum(var, 1e-6)
        
        with tf.control_dependencies([
                tf.debugging.assert_positive(var)]):
            alpha = mean*(mean*(1-mean)/var-1)
            beta = (1-mean)*(mean*(1-mean)/var-1)

        alpha = tf.debugging.assert_all_finite(alpha, "alpha not finite")
        beta = tf.debugging.assert_all_finite(beta, "beta not finite")
        
        return alpha, beta

    def uniform():
        alpha = tf.constant(1.0, tf.float32)
        beta = tf.constant(1.0, tf.float32)
        return alpha, beta
    
    return tf.cond(sumw > 0, lambda: mm(data, weights, sumw), uniform)
