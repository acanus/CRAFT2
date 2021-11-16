from lib import *
from net import *
from loss import *
import matplotlib.pyplot as plt
plt.ion()

# from net import *
from datagen import *
print(os.environ.get('BATCH_SIZE'))
print(os.environ.get('DATASET_PATH'))
print(os.environ.get('MODEL_NAME'))
parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type = int, default = 768) # kích thước đầu vào để đào tạo mạng
parser.add_argument('--batch_size', type = int, default = os.environ.get('BATCH_SIZE')) # kích thước lô để đào tạo
parser.add_argument('--init_learning_rate', type = float, default = 0.1) # tốc độ học ban đầu
parser.add_argument('--lr_decay_rate', type = float, default = 0.9) # tỷ lệ phân rã cho tỷ lệ học tập
parser.add_argument('--lr_decay_steps', type = int, default = 25) # số bước mà sau đó tốc độ học được giảm dần theo tốc độ giảm dần
parser.add_argument('--epochs', type = int, default = 20000) # số kỷ nguyên tối đa
parser.add_argument('--checkpoint_path', type = str, default = 'tmp/checkpoint') # # đường dẫn đến một thư mục để lưu các điểm kiểm tra của mô hình trong quá trình đào tạo
parser.add_argument('--gpu_list', type = str, default = '0')  # Danh sách gpu để sử dụng
parser.add_argument('--model_name', type = str, default = os.environ.get('MODEL_NAME'))  # chọn model train
# parser.add_argument('--model_name', type = str, default = 'vgg16')  # chọn model train
parser.add_argument('--training_data_path', type = str, default = os.environ.get('DATASET_PATH')) # đường dẫn đến training data
parser.add_argument('--suppress_warnings_and_error_messages', type = bool, default = True) # có hiển thị thông báo lỗi và cảnh báo trong quá trình đào tạo hay không (một số thông báo lỗi trong quá trình đào tạo dự kiến ​​sẽ xuất hiện do cách tạo các bản vá lỗi cho quá trình đào tạo)
parser.add_argument('--load_weight', type = bool, default = True)
parser.add_argument('--test_dir', type = str, default = 'images')

parser.add_argument('--vis', type = bool, default = True)
parser.add_argument('--vis_num_batch_size', type = bool, default = 50) # cứ sao 50 batch size sẽ hiển thị kết quả train 1 lần

FLAGS = parser.parse_args()

# def lr_decay(epoch):
#     return FLAGS.init_learning_rate * np.power(FLAGS.lr_decay_rate, epoch // FLAGS.lr_decay_steps)

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, boundaries, learning_rate):
        self.boundaries = boundaries
        self.learning_rate = learning_rate

    def __call__(self, step):
        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.boundaries, self.learning_rate)
        return  learning_rate_fn(step)

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self,test_generator) -> None:
        super().__init__()
        self.test_generator=test_generator
        self.num_samples = len(test_generator)
        self.fig, (self.ax1, self.ax2,self.ax3,self.ax4,self.ax5) = plt.subplots(1, 5,figsize=(12, 10))
    def on_batch_end(self, batch, logs=None):
        if batch % 200 == 0:
            data=self.test_generator.__getitem__(random.randint(0,self.num_samples-1))
            gt= data[1][0]
            image = np.expand_dims(data[0][0],0)
            result=self.model.predict(image)
            self.ax1.imshow(image[0].astype('uint8'))
            self.ax2.imshow(result[0][:,:,0])
            self.ax3.imshow(result[0][:,:,1])
            self.ax4.imshow(gt[:,:,0])
            self.ax5.imshow(gt[:,:,1])
            self.ax1.set_title('Min: '+str(np.min(image[0]))+' Max: '+str(np.max(image[0])))
            self.ax2.set_title('Min: '+str(np.min(result[0][:,:,0]))+' Max: '+str(np.max(result[0][:,:,0])))
            self.ax3.set_title('Min: '+str(np.min(result[0][:,:,1]))+' Max: '+str(np.max(result[0][:,:,1])))
            self.ax4.set_title('Ground Truth 1')
            self.ax5.set_title('Ground Truth 2')
            plt.draw()
            plt.show(block=False)
            plt.pause(.001)
def TestGenerator(test_dir):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(FLAGS.input_size, FLAGS.input_size),
        batch_size=1,
        shuffle=False,
        class_mode=None,
        color_mode='rgb',
        interpolation='bilinear')
    return test_generator
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list            
    # test_generator = TestGenerator(FLAGS.test_dir) 
    if not os.path.exists(FLAGS.checkpoint_path):
        os.mkdir(FLAGS.checkpoint_path)

    train_data_generator = SynthTextDataGeneratorUpdate(FLAGS.training_data_path, (FLAGS.input_size,FLAGS.input_size), FLAGS.batch_size)
    train_steps = len(train_data_generator)
    craft = get_model(FLAGS.model_name)
    checkpoint_path = os.path.sep.join([FLAGS.checkpoint_path, "model_craft_%s-{epoch:04d}.ckpt"%(FLAGS.model_name)])
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decay)
    modelckpt = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_freq = 100 * FLAGS.batch_size,  save_weights_only = True, verbose = 1)
    craft.save_weights(checkpoint_path.format(epoch = 0))
    # callbacks = [lr_scheduler,modelckpt]
    vis_callback = MyCallback(test_generator = train_data_generator)
    callbacks = [modelckpt,vis_callback]
    optimizer = tf.keras.optimizers.Adam(learning_rate = MyLRSchedule([50000, 200000], [FLAGS.init_learning_rate, FLAGS.init_learning_rate / 10. , FLAGS.init_learning_rate / 100.]))
    craft.compile(optimizer = optimizer,loss= MSE_OHEM_Loss,  run_eagerly = True)
    # craft.compile(optimizer = optimizer)
    #craft.build(input_shape=(FLAGS.input_size,FLAGS.input_size,3))

    # if(FLAGS.load_weight == True):
    #     #build model by fiting 1 epoch (to load weights)
    #     H = craft.fit(train_data_generator,
    #                 steps_per_epoch = 1,
    #                 batch_size = FLAGS.batch_size,
    #                 epochs = 1)
    #     craft.load_weights(os.path.join(FLAGS.checkpoint_path,"model_craft_resnet50-0001.ckpt"))

    # Khôi phục lại trọng số mạng để train tiếp
    if(FLAGS.load_weight == True):
        craft.load_weights(latest)
        
    # Huấn luyện mạng
    print("[INFO] Huấn luyện mạng...")
    H = craft.fit(train_data_generator,
                steps_per_epoch = train_steps,
                batch_size = FLAGS.batch_size,
                epochs = FLAGS.epochs,
                callbacks = callbacks)
if __name__ == '__main__':
    main()