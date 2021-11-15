from lib import *
from loss import mse, MSE_OHEM_Loss,weighted_bce

def upconv(input, num_filters):
    x = tf.keras.layers.Conv2D(num_filters[0], 1, activation = "relu", padding = "same")(input)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(num_filters[1], 3, activation = "relu", padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def upsample(x):
    x = tf.keras.layers.UpSampling2D((2, 2), interpolation = "bilinear")(x)
    return x

def Conv_cls(input_tensor, num_class):

    x = tf.keras.layers.Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu')(input_tensor)
    x = tf.keras.layers.Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(16, kernel_size = 3, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(16, kernel_size = 1, padding = 'same', activation = 'relu')(x)
    x = tf.keras.layers.Conv2D(num_class, kernel_size = 1, padding = 'same', activation = 'sigmoid')(x)

    return x

def get_model(model_name):

    if model_name == "vgg16":
        input_image = tf.keras.layers.Input(shape = (None, None, 3), name = 'input_image')
        
        """ Pre-trained VGG16 Model """
        vgg16 = tf.keras.applications.vgg16.VGG16(input_tensor = input_image, weights = 'imagenet', include_top = False, pooling = None)
        vgg16.trainable = False

        # VGG end
        # tage 6
        source = vgg16.get_layer('block5_conv3').output
        # tage 5
        x = tf.keras.layers.MaxPooling2D(3, strides = 1, padding = 'same', name = 'block5_pool')(source)                         # w/16 512
        x = tf.keras.layers.Conv2D(1024, kernel_size = 3, activation = "relu", padding = 'same', dilation_rate = 6)(x)           # w/16 1024
        x = tf.keras.layers.Conv2D(1024, kernel_size = 1, activation = "relu", padding = "same")(x)                              # w/16 1024
        
        # U-net start
        x = tf.keras.layers.Concatenate()([x, source])      # w/16 1024 + 512
        x = upconv(x, [512, 256])           # w/16 256
        x = upsample(x) # w/8 256

        x = tf.keras.layers.Concatenate()([x, vgg16.get_layer('block4_conv3').output])      # w/8 256 + 512
        x = upconv(x, [256, 128])           # w/8 128
        x = upsample(x)                     # w/4 128

        x = tf.keras.layers.Concatenate()([x, vgg16.get_layer('block3_conv3').output])     # w/4 128 + 256
        x = upconv(x, [128, 64])           # w/4 64
        x = upsample(x)                    # w/2 64

        x = tf.keras.layers.Concatenate()([x, vgg16.get_layer('block2_conv2').output])    # w/2 64 + 128
        x = upconv(x, [64, 32])            # w/2 64
        # U-net end
        
        # feature
        output = Conv_cls(x, 2)

        # # region score
        
        # region_score = Conv2D(1, (1, 1), activation = tf.nn.sigmoid, name = 'region_score')(x)

        # # affinity score
        # affinity_score = Conv2D(1, (1, 1), activation = tf.nn.sigmoid, name = 'affinity_score')(x)

        # output = concatenate([region_score, affinity_score], axis = 3, name = 'predict_scores')
        preprocess_layer = tf.keras.applications.vgg16.preprocess_input
        base_model = tf.keras.models.Model(inputs = input_image, outputs = output, name = 'vgg16_unet_base')
        input = tf.keras.layers.Input(shape = (None, None, 3), name = 'main_image')
        model = tf.keras.models.Model(inputs = input, outputs = base_model(preprocess_layer(input)), name = 'vgg16_unet')
        # model = CRAFT_model(inputs = input_image, outputs = output, name = 'vgg16_unet')
        
        return model

    elif model_name == "resnet50":
        input_image = tf.keras.layers.Input(shape = (None, None, 3), name = 'input_image')
        
        """ Pre-trained VGG16 Model """
        resnet50 = tf.keras.applications.resnet50.ResNet50(input_tensor = input_image, weights = 'imagenet', include_top = False, pooling = None)
        resnet50.trainable = False

        # resnet50 end
        # tage 6
        source = resnet50.get_layer('conv5_block3_3_conv').output

        # tage 5
        x = tf.keras.layers.MaxPooling2D(3, strides = 1, padding = 'same', name = 'block5_pool')(source)                         # w/32 512
        x = tf.keras.layers.Conv2D(512, kernel_size = 3, activation = "relu", padding = 'same', dilation_rate = 6)(x)           # w/32 512
        x = tf.keras.layers.Conv2D(512, kernel_size = 1, activation = "relu", padding = "same")(x)                              # w/32 512

        # U-net start
        x = tf.keras.layers.Concatenate()([x, source])      # w/32 2048 + 512
        x = upconv(x, [512, 256])           # w/32 256
        x = upsample(x)                     # w/16 256

        x = tf.keras.layers.Concatenate()([x, resnet50.get_layer('conv4_block6_3_conv').output])      # w/16 256 + 1024
        x = upconv(x, [256, 128])           # w/16 128
        x = upsample(x)                     # w/8 128

        x = tf.keras.layers.Concatenate()([x, resnet50.get_layer('conv3_block4_3_conv').output])     # w/8 128 + 512
        x = upconv(x, [128, 64])           # w/8 64
        x = upsample(x)                    # w/4 64

        x = tf.keras.layers.Concatenate()([x, resnet50.get_layer('conv2_block3_3_conv').output])    # w/4 64 + 256
        x = upconv(x, [64, 32])            # w/4 64
        x = upsample(x)                    # w/2 32

        # feature
        output = Conv_cls(x, 2)

        preprocess_layer = tf.keras.applications.resnet50.preprocess_input
        base_model = tf.keras.models.Model(inputs = input_image, outputs = output, name = 'resnet50_unet_base')
        input = tf.keras.layers.Input(shape = (None, None, 3), name = 'main_image')
        model = tf.keras.models.Model(inputs = input, outputs = base_model(preprocess_layer(input)), name = 'resnet50_unet')
        # model = CRAFT_model(inputs = input_image, outputs = output, name = 'resnet50_unet')
        return model
    elif model_name == "mobilenet":
        input_image = tf.keras.layers.Input(shape = (None, None, 3), name = 'input_image')
        
        """ Pre-trained VGG16 Model """
        mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(input_tensor = input_image, weights = 'imagenet', include_top = False, pooling = None)
        mobilenet.trainable = False

        # resnet50 end
        # tage 6
        source = mobilenet.get_layer('out_relu').output

        # tage 5
        x = tf.keras.layers.MaxPooling2D(3, strides = 1, padding = 'same', name = 'block5_pool')(source)                         # w/32 512
        x = tf.keras.layers.Conv2D(512, kernel_size = 3, activation = "relu", padding = 'same', dilation_rate = 6)(x)           # w/32 512
        x = tf.keras.layers.Conv2D(512, kernel_size = 1, activation = "relu", padding = "same")(x)                              # w/32 512

        # U-net start
        x = tf.keras.layers.Concatenate()([x, source])      # w/32 2048 + 512
        x = upconv(x, [512, 256])           # w/32 256
        x = upsample(x)                     # w/16 256

        x = tf.keras.layers.Concatenate()([x, mobilenet.get_layer('block_13_expand').output])      # w/16 256 + 1024
        x = upconv(x, [256, 128])           # w/16 128
        x = upsample(x)                     # w/8 128

        x = tf.keras.layers.Concatenate()([x, mobilenet.get_layer('block_6_expand').output])     # w/8 128 + 512
        x = upconv(x, [128, 64])           # w/8 64
        x = upsample(x)                    # w/4 64

        x = tf.keras.layers.Concatenate()([x, mobilenet.get_layer('block_3_expand').output])    # w/4 64 + 256
        x = upconv(x, [64, 32])            # w/4 64
        x = upsample(x)                    # w/2 32

        # feature
        output = Conv_cls(x, 2)

        preprocess_layer = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.models.Model(inputs = input_image, outputs = output, name = 'mobilenet_unet_base')
        input = tf.keras.layers.Input(shape = (None, None, 3), name = 'main_image')
        model = tf.keras.models.Model(inputs = input, outputs = base_model(preprocess_layer(input)), name = 'mobilenet_unet')
        # model = CRAFT_model(inputs = input_image, outputs = output, name = 'resnet50_unet')
        return model
    elif model_name == "mobilenet_unet":
        input_image = tf.keras.layers.Input(shape = (None, None, 3), name = 'input_image')

        encoder = tf.keras.applications.mobilenet_v2.MobileNetV2(input_tensor = input_image,  weights="imagenet", include_top=False, alpha=0.35)
        encoder.trainable=False
        skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
        encoder_output = encoder.get_layer("block_13_expand_relu").output  
        f = [16, 32, 48, 64]
        x = encoder_output
        for i in range(1, len(skip_connection_names), 1):
            x_skip = encoder.get_layer(skip_connection_names[-i]).output
            x = tf.keras.layers.UpSampling2D((2, 2))(x)
            x = tf.keras.layers.Concatenate()([x, x_skip])
            
            x = tf.keras.layers.Conv2D(f[-i], (3, 3), padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            
            x = tf.keras.layers.Conv2D(f[-i], (3, 3), padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            
        x = tf.keras.layers.Conv2D(2, (1, 1), padding="same")(x)
        output = tf.keras.layers.Activation("sigmoid",name="main_output")(x)
        
        preprocess_layer = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.models.Model(inputs = input_image, outputs = output, name = 'mobilenet_unet_base')
        input = tf.keras.layers.Input(shape = (None, None, 3), name = 'main_image')
        model = tf.keras.models.Model(inputs = input, outputs = base_model(preprocess_layer(input)), name = 'mobilenet_unet')
        return model

class CRAFT_model(tf.keras.Model):
    def __init__(self, input_size = 512, model_name = "vgg16", **kwargs):
        super(CRAFT_model, self).__init__(**kwargs)

        self.compiled_loss = MSE_OHEM_Loss    
        self.model = get_model(model_name)

    def train_step(self, data):
        input_images, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self(input_images)
            loss = MSE_OHEM_Loss(y_true, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'loss': loss}

    def call(self, inputs):
        if isinstance(inputs, tuple):
            return self.model(inputs[0])
        else:
            return self.model(inputs)


