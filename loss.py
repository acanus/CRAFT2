from lib import *

def ohem(loss, fg_mask, bg_mask, negative_ratio = 3.):
    fg_num = tf.math.reduce_sum(fg_mask)
    bg_num = tf.math.reduce_sum(bg_mask)

    neg_num = tf.math.maximum(tf.cast(fg_num * negative_ratio, dtype = tf.int32), tf.constant(10000, dtype = tf.int32))
    neg_num = tf.math.minimum(tf.cast(bg_num, dtype = tf.int32), neg_num)

    neg_loss = loss * bg_mask
    vals, _ = tf.math.top_k(tf.reshape(neg_loss, shape = [-1]), k = neg_num)
    bg_bool_mask = tf.cast(bg_mask, dtype = tf.bool)
    hard_bg_bool_mask = tf.math.logical_and(bg_bool_mask, tf.math.greater_equal(neg_loss, vals[-1]))
    hard_bg_mask = tf.cast(hard_bg_bool_mask, dtype = tf.float32)

    return hard_bg_mask

def batch_ohem(loss, fg_mask, bg_mask, negative_ratio = 3.):
    return tf.map_fn(lambda x: ohem(x[0], x[1], x[2], negative_ratio), elems = [loss, fg_mask, bg_mask], dtype = tf.float32)

def huber_loss(y_true, y_pred, delta = 0.5):
    residual = tf.math.abs(y_true - y_pred)
    large_loss = 0.5 * tf.math.pow(y_true - y_pred, 2)
    small_loss = delta * residual - 0.5 * tf.math.square(delta)
    return tf.where(tf.math.less(residual, delta), large_loss, small_loss)

def craft_huber_loss(region_true, affinity_true, region_pred, affinity_pred, confidence, fg_mask, bg_mask):
    region_true, affinity_true, region_pred, affinity_pred, confidence, fg_mask, bg_mask = region_true, affinity_true, region_pred, affinity_pred, confidence, fg_mask, bg_mask
    eps = 1e-5
    confidence = confidence

    l_region = huber_loss(region_true, region_pred)
    l_region = l_region * confidence

    l_affinity = huber_loss(affinity_true, affinity_pred)
    l_affinity = l_affinity * confidence

    l_total = l_region + l_affinity

    # hard_bg_mask = ohem(l_total, fg_mask, bg_mask)
    hard_bg_mask = batch_ohem(l_total, fg_mask, bg_mask)
    # hard_bg_mask = bg_mask
    train_mask = hard_bg_mask + fg_mask
    l_total = l_total * train_mask

    return tf.math.reduce_sum(l_total) / (tf.math.reduce_sum(confidence * train_mask) + eps)