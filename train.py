from lib import *
from net import CRAFT_model
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
parser.add_argument('--load_weight', type = bool, default = False)
parser.add_argument('--test_dir', type = str, default = 'images')
FLAGS = parser.parse_args()

def lr_decay(epoch):
    return FLAGS.init_learning_rate * np.power(FLAGS.lr_decay_rate, epoch // FLAGS.lr_decay_steps)
class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self,test_generator) -> None:
        super().__init__()
        self.test_generator=test_generator
        self.fig, (self.ax1, self.ax2,self.ax3,self.ax4,self.ax5) = plt.subplots(1, 5,figsize=(12, 10))
    def on_batch_end(self, batch, logs=None):
        if batch % 50 == 0:
            gt= self.test_generator.__getitem__(10)[1][0]
            data = np.expand_dims(self.test_generator.__getitem__(10)[0][0],0)
            result=self.model.predict(data)
            self.ax1.imshow(data[0].astype('uint8'))
            self.ax2.imshow(result[0][:,:,0])
            self.ax3.imshow(result[0][:,:,1])
            self.ax4.imshow(gt[:,:,0])
            self.ax5.imshow(gt[:,:,1])
            self.ax1.set_title('Min: '+str(np.min(data[0]))+' Max: '+str(np.max(data[0])))
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
    test_generator = TestGenerator(FLAGS.test_dir) 
     # kiểm tra xem đường dẫn điểm kiểm tra có tồn tại không
    if not os.path.exists(FLAGS.checkpoint_path):
        os.mkdir(FLAGS.checkpoint_path)
    train_data_generator = SynthTextDataGenerator(FLAGS.training_data_path, (FLAGS.input_size,FLAGS.input_size), FLAGS.batch_size)
    print('đào tạo tổng số lô mỗi kỷ nguyên : {}'.format(len(train_data_generator)))
    # Khởi tạo mạng nơ-ron
    print("[INFO] Biên dịch mô hình...")
    craft = CRAFT_model(FLAGS.input_size, FLAGS.model_name)
    # craft = get_model('vgg16')

    # tạo đường dẫn lưu file
    checkpoint_path = os.path.sep.join([FLAGS.checkpoint_path, "model_craft_%s-{epoch:04d}.ckpt"%(FLAGS.model_name)])

    # tạo kiểm soát mô hình
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decay)
    modelckpt = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_freq = 1000 * FLAGS.batch_size,  save_weights_only = True, verbose = 1)
    
    # Lưu trọng số bằng định dạng 'checkpoint_path'
    #craft.save_weights(checkpoint_path.format(epoch = 0))
    
    # Hàm callbacks
    callbacks = [lr_scheduler,modelckpt]

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(FLAGS.init_learning_rate)

    # Complie model
    print("[INFO] Biên dịch mô hình...")
    craft.compile(optimizer = optimizer, run_eagerly = True)
    # craft.compile(optimizer = optimizer)
    #craft.build(input_shape=(FLAGS.input_size,FLAGS.input_size,3))

    if(FLAGS.load_weight == True):
        #build model by fiting 1 epoch (to load weights)
        H = craft.fit(train_data_generator,
                    steps_per_epoch = 1,
                    batch_size = FLAGS.batch_size,
                    epochs = 1)
        craft.load_weights(os.path.join(FLAGS.checkpoint_path,"model_craft_resnet50-0001.ckpt"))
    # Huấn luyện mạng
    print("[INFO] Huấn luyện mạng...")
    fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5,figsize=(12, 10))
    for epoch in range(FLAGS.epochs):
        H = craft.fit(train_data_generator,
                    steps_per_epoch = None,
                    batch_size = None,
                    epochs = 1,
                    callbacks = callbacks)
        print('Epoch: {}/{}'.format(epoch + 1, FLAGS.epochs))
        data=train_data_generator.__getitem__(4)
        gt= data[1][0]
        image = np.expand_dims(data[0][0],0)
        result=craft.predict(image.astype('float32'))
        ax1.imshow(image[0].astype('uint8'))
        ax2.imshow(result[0][:,:,0])
        ax3.imshow(result[0][:,:,1])
        ax4.imshow(gt[:,:,0])
        ax5.imshow(gt[:,:,1])
        ax1.set_title('Min: '+str(np.min(image[0]))+' Max: '+str(np.max(image[0])))
        ax2.set_title('Min: '+str(np.min(result[0][:,:,0]))+' Max: '+str(np.max(result[0][:,:,0])))
        ax3.set_title('Min: '+str(np.min(result[0][:,:,1]))+' Max: '+str(np.max(result[0][:,:,1])))
        ax4.set_title('Ground Truth 1')
        ax5.set_title('Ground Truth 2')
        plt.draw()
        plt.show(block=False)
        plt.pause(.001)
    #craft.save_weights(checkpoint_path.format(epoch = epoch + 1))
    # H = craft.fit(train_data_generator,
    #             steps_per_epoch = None,
    #             batch_size = None,
    #             epochs = FLAGS.max_epochs,
    #             # validation_data = valid_data_generator,
    #             # validation_steps = valid_steps,
    #             callbacks = callbacks)
    
    # lưu lại lịch sử đào tạo
    plt.figure(figsize = (10, 6))
    plt.plot(H.history['loss'], color = 'black')
    plt.title('model_craft_%s Loss'%(FLAGS.model_name))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss'], loc = 'upper right')
    plt.grid()
    plt.savefig('model_craft_%s.png'%(FLAGS.model_name), dpi = 480, bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    main()