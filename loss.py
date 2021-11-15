from lib import *
def weighted_bce(y_true, y_pred):
    loss_every_sample = []
    batch_size = y_true.get_shape().as_list()[0]
    for i in range(batch_size):
        weights = (y_true[i]*50) + 1.       
        bce = tf.keras.losses.binary_crossentropy(y_true[i], y_pred[i])
        weighted_bce = tf.math.reduce_mean(tf.math.multiply(weights[:,:,0],bce[0]))
        weighted_bce1 = tf.math.reduce_mean(tf.math.multiply(weights[:,:,1],bce[1]))
        loss_every_sample.append(weighted_bce*0.5+weighted_bce1*0.5)
    return tf.math.reduce_mean(tf.convert_to_tensor(loss_every_sample))
def MSE_OHEM_Loss(y_true,y_pred):
    loss_every_sample = []
    batch_size = y_true.get_shape().as_list()[0]
    for i in range(batch_size):
        output_img = tf.reshape(y_pred[i], [-1])
        target_img = tf.reshape(y_true[i], [-1])
        positive_mask = tf.cast(tf.greater(target_img, 0), dtype = tf.float32)
        sample_loss = tf.math.square(tf.math.subtract(output_img, target_img))
        
        num_all = output_img.get_shape().as_list()[0]
        num_positive = tf.cast(tf.math.reduce_sum(positive_mask), dtype = tf.int32)
        
        positive_loss = tf.math.multiply(sample_loss, positive_mask)
        positive_loss_m = tf.math.reduce_sum(positive_loss)/tf.cast(num_positive, dtype = tf.float32)
        nagative_loss = tf.math.multiply(sample_loss, (1 - positive_mask))
        # nagative_loss_m = tf.math.reduce_sum(nagative_loss)/(num_all - num_positive)

        k = num_positive * 3     
        k = tf.cond((k + num_positive) > num_all, lambda: tf.cast((num_all - num_positive), dtype = tf.int32), lambda: k)
        k = tf.cond(k > 0, lambda: k, lambda: k + 1)   
        nagative_loss_topk, _ = tf.math.top_k(nagative_loss, k)
        res = tf.cond(k < 10, lambda: tf.math.reduce_mean(sample_loss),
                              lambda: positive_loss_m + tf.math.reduce_sum(nagative_loss_topk)/tf.cast(k, dtype=tf.float32))
        loss_every_sample.append(res)

    return tf.math.reduce_mean(tf.convert_to_tensor(loss_every_sample))

def mse(y_true,y_pred): # vì dự liệu là chính xác từng ký tự nên confidence = 1... ta áp dụng tính mse
    loss_every_sample = []
    batch_size = y_true.get_shape().as_list()[0]
    for i in range(batch_size):
        output_img = tf.reshape(y_pred[i], [-1])
        target_img = tf.reshape(y_true[i], [-1])
        loss = tf.math.square(tf.math.subtract(target_img, output_img))
        loss_every_sample.append(tf.math.reduce_mean(loss))

    return tf.math.reduce_mean(tf.convert_to_tensor(loss_every_sample))

# if __name__ == '__main__':
#     output_imgs = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)
#     output_imgs = tf.reshape(output_imgs, (2, 1, 2, 2))
#     # target_imgs = tf.constant([[[1.1, 2.1], [3.5, 4.1]], [[5.2, 6.5], [7.6, 8.5]]], dtype=tf.float32)
#     target_imgs = tf.constant([[[25, 25], [30, 35]], [[40, 42], [50, 53]]], dtype=tf.float32)
#     target_imgs = tf.reshape(target_imgs, (2, 1, 2, 2))
    
#     m = mse(output_imgs, target_imgs)
#     print(m)
#     r = MSE_OHEM_Loss(output_imgs, target_imgs)
#     print(r)