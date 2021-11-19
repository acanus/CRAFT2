from lib import *
from loss import craft_huber_loss
from img_util import img_unnormalize

class UpsampleLike(tf.keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor."""
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = tf.keras.backend.shape(target)
        if tf.keras.backend.image_data_format() == 'channels_first':
            source = tf.transpose(source, (0, 2, 3, 1))
            output = tf.image.resize(source, (target_shape[2], target_shape[3]), method='nearest')
            output = tf.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return tf.image.resize(source, (target_shape[1], target_shape[2]), method='nearest')

def upconv(input, num_filters):
    x = tf.keras.layers.Conv2D(num_filters[0], 1, activation = "relu", padding = "same")(input)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(num_filters[1], 3, activation = "relu", padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def Conv_cls(input_tensor, num_class):
    x = tf.keras.layers.Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu')(input_tensor)
    x = tf.keras.layers.Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(16, kernel_size = 3, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(16, kernel_size = 1, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(16, kernel_size = num_class, padding = 'same', activation = 'sigmoid')(x)

    return x
    
def get_model(model_name):

    if model_name == "vgg16":
        input_image = tf.keras.layers.Input(shape = (None, None, 3), name = 'input_image')
        # x = tf.keras.layers.experimental.preprocessing.Rescaling(scale = 1. / 255.0, offset = 0.0)(input_image)
        
        """ Pre-trained VGG16 Model """
        vgg16 = tf.keras.applications.vgg16.VGG16(input_tensor = input_image, weights = 'imagenet', include_top = False, pooling = None)
        vgg16.trainable = False

        # VGG end
        # tage 6
        source = vgg16.get_layer('block5_conv3').output

        # tage 5
        x = tf.keras.layers.MaxPooling2D(3, strides = 1, padding = 'same', name = 'block5_pool')(source)    # w/16 512
        x = tf.keras.layers.Conv2D(1024, kernel_size = 3, activation = "relu", padding = 'same', dilation_rate = 6)(x)           # w/16 1024
        x = tf.keras.layers.Conv2D(1024, kernel_size = 1, activation = "relu", padding = "same")(x)                              # w/16 1024
        
        # U-net start
        x = UpsampleLike(name = 'resize_1')([x, source])
        x = tf.keras.layers.Concatenate()([x, source])      # w/16 1024 + 512
        x = upconv(x, [512, 256])           # w/8 256

        source = vgg16.get_layer('block4_conv3').output
        x = UpsampleLike(name = 'resize_2')([x, source])
        x = tf.keras.layers.Concatenate()([x, source])      # w/8 256 + 512
        x = upconv(x, [256, 128])           # w/4 128

        source = vgg16.get_layer('block3_conv3').output
        x = UpsampleLike(name = 'resize_3')([x, source])
        x = tf.keras.layers.Concatenate()([x, source])     # w/4 128 + 256
        x = upconv(x, [128, 64])           # w/2 64

        source = vgg16.get_layer('block2_conv2').output
        x = UpsampleLike(name = 'resize_4')([x, source])
        x = tf.keras.layers.Concatenate()([x, source])    # w/2 64 + 128
        x = upconv(x, [64, 32])           # w/2 32
        # U-net end
        
        # feature
        x = Conv_cls(x, 2)

        # region score
        region_score = tf.keras.layers.Lambda(lambda layer: layer[:, :, :, 0])(x)

        # affinity score
        affinity_score = tf.keras.layers.Lambda(lambda layer: layer[:, :, :, 1])(x)

        model = tf.keras.models.Model(inputs = input_image, outputs = [region_score, affinity_score], name = 'vgg16_unet')

        return model

    elif model_name == "resnet50":
        input_image = tf.keras.layers.Input(shape = (None, None, 3), name = 'input_image')
        # x = tf.keras.layers.experimental.preprocessing.Rescaling(scale = 1. / 255.0, offset = 0.0)(input_image)

        """ Pre-trained ResNet50 Model """
        resnet50 = tf.keras.applications.resnet50.ResNet50(input_tensor = input_image, weights = 'imagenet', include_top = False, pooling = None)
        resnet50.trainable = False
        # tage 6
        source = resnet50.get_layer('conv5_block3_3_conv').output

        # tage 5
        x = tf.keras.layers.MaxPooling2D(3, strides = 1, padding = 'same', name = 'res5c_pool')(source)
        x = tf.keras.layers.Conv2D(512, kernel_size = 3, padding = 'same', dilation_rate = 6)(x)
        x = tf.keras.layers.Conv2D(512, kernel_size = 1)(x)

        # tage 5 + tage 6 = output1
        x = UpsampleLike(name = 'resize_1')([x, source])
        x = tf.keras.layers.concatenate([x, source], axis = 3)
        x = upconv(x, [512, 256])

        # tage 4
        source = resnet50.get_layer('conv4_block6_3_conv').output

        # tag 4 + output1 = output2
        x = UpsampleLike(name = 'resize_2')([x, source])
        x = tf.keras.layers.concatenate([x, source], axis = 3)
        x = upconv(x, [256, 128])

        # tage 3
        source = resnet50.get_layer('conv3_block4_3_conv').output

        # tag 3 + output2 = output3
        x = UpsampleLike(name = 'resize_3')([x, source])
        x = tf.keras.layers.concatenate([x, source], axis = 3)
        x = upconv(x, [128, 64])
        
        # tage 2
        source = resnet50.get_layer('conv2_block3_3_conv').output

        # tag 2 + output3 = output4
        x = UpsampleLike(name = 'resize_4')([x, source])
        x = tf.keras.layers.concatenate([x, source], axis = 3)
        x = upconv(x, [64, 32])
        x = tf.keras.layers.UpSampling2D(size = (2, 2), interpolation = 'bilinear', data_format = 'channels_last', name = 'resize_5')(x)

        # feature
        x = Conv_cls(x, 2)

        # region score
        region_score = tf.keras.layers.Lambda(lambda layer: layer[:, :, :, 0])(x)

        # affinity score
        affinity_score = tf.keras.layers.Lambda(lambda layer: layer[:, :, :, 1])(x)

        model = tf.keras.models.Model(inputs = input_image, outputs = [region_score, affinity_score], name = 'resnet50_unet')
        
        return model

class CRAFT_model(tf.keras.Model):
    def __init__(self, model_name = "vgg16", vis = False, num_batch_size = 50, **kwargs):
        super(CRAFT_model, self).__init__(**kwargs)
        self.vis = vis
        self.compiled_loss = craft_huber_loss    
        self.model = get_model(model_name)
        self.vis_num_batch_size = num_batch_size
        self.num_batch_size = 0
        self.fig = plt.subplots(1, 5, figsize = (12, 10)) if vis else None

    def train_step(self, data):
        [input_images], [region_scores, affinity_scores, confidence_scores, fg_masks, bg_masks] = data
        self.num_batch_size += 1
        with tf.GradientTape() as tape:
            region_preds, affinity_preds = self(input_images)
            loss = craft_huber_loss(region_scores, affinity_scores, region_preds, affinity_preds, confidence_scores, fg_masks, bg_masks)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if self.vis:
            if self.num_batch_size == self.vis_num_batch_size:
                with tf.experimental.async_scope():
                    self.__vis_data_train__(input_images[0].numpy(), region_scores[0].numpy(), affinity_scores[0].numpy(), region_preds[0].numpy(), affinity_preds[0].numpy())
                    self.num_batch_size = 0

        return {'loss': loss}

    def call(self, inputs):
        if isinstance(inputs, tuple):
            return self.model(inputs[0])
        else:
            return self.model(inputs)
 
    def __vis_data_train__(self, image, region_gt, affinity_gt, region_pred, affinity_pred):      
        self.fig[1][0].imshow(img_unnormalize(image).astype('uint8'))
        self.fig[1][1].imshow(region_pred)
        self.fig[1][2].imshow(affinity_pred)
        self.fig[1][3].imshow(region_gt)
        self.fig[1][4].imshow(affinity_gt)
        self.fig[1][0].set_title('Min: '+str(np.min(image))+' Max: '+str(np.max(image)))
        self.fig[1][1].set_title('Min: '+str(np.min(region_pred))+' Max: '+str(np.max(region_pred)))
        self.fig[1][2].set_title('Min: '+str(np.min(affinity_pred))+' Max: '+str(np.max(affinity_pred)))
        self.fig[1][3].set_title('Ground Truth 1')
        self.fig[1][4].set_title('Ground Truth 2')
        plt.draw()
        plt.show(block = False)
        plt.pause(.001)