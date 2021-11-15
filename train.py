from lib import *
from net import CRAFT_model
from datagen import *

plt.ion()

parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type = int, default = 512) # kích thước đầu vào để đào tạo mạng
parser.add_argument('--batch_size', type = int, default = 2) # kích thước lô để đào tạo
parser.add_argument('--init_learning_rate', type = float, default = 0.001) # tốc độ học ban đầu
parser.add_argument('--epochs', type = int, default = 200000) # số kỷ nguyên tối đa
parser.add_argument('--checkpoint_path', type = str, default = 'tmp/checkpoint') # # đường dẫn đến một thư mục để lưu các điểm kiểm tra của mô hình trong quá trình đào tạo
parser.add_argument('--gpu_list', type = str, default = '0')  # Danh sách gpu để sử dụng
parser.add_argument('--model_name', type = str, default = "resnet50")  # chọn model train
# parser.add_argument('--model_name', type = str, default = 'vgg16')  # chọn model train
parser.add_argument('--training_data_path', type = str, default = r"datasets\synthtext\SynthText") # đường dẫn đến training data
parser.add_argument('--suppress_warnings_and_error_messages', type = bool, default = True) # có hiển thị thông báo lỗi và cảnh báo trong quá trình đào tạo hay không (một số thông báo lỗi trong quá trình đào tạo dự kiến ​​sẽ xuất hiện do cách tạo các bản vá lỗi cho quá trình đào tạo)
parser.add_argument('--load_weight', type = bool, default = False)
parser.add_argument('--test_dir', type = str, default = 'images')
parser.add_argument('--vis', type = bool, default = False)
FLAGS = parser.parse_args()

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, boundaries, learning_rate):
        self.boundaries = boundaries
        self.learning_rate = learning_rate

    def __call__(self, step):
        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.boundaries, self.learning_rate)
        return  learning_rate_fn(step)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

     # kiểm tra xem đường dẫn điểm kiểm tra có tồn tại không
    if not os.path.exists(FLAGS.checkpoint_path):
        os.mkdir(FLAGS.checkpoint_path)
    
    # Tạo data train
    train_data_generator = SynthTextDataGenerator(FLAGS.training_data_path, (FLAGS.input_size, FLAGS.input_size), FLAGS.batch_size)
    train_steps = len(train_data_generator)
    print('đào tạo tổng số lô mỗi kỷ nguyên : {}'.format(train_steps))

    # Khởi tạo mạng nơ-ron
    print("[INFO] Biên dịch mô hình...")
    craft = CRAFT_model(FLAGS.model_name, vis = FLAGS.vis)

    # tạo đường dẫn lưu file
    checkpoint_path = os.path.sep.join([FLAGS.checkpoint_path, "model_craft_%s-{epoch:04d}.ckpt"%(FLAGS.model_name)])
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # tạo kiểm soát mô hình
    modelckpt = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_freq = 50 * FLAGS.batch_size,  save_weights_only = True, verbose = 1)

    # Lưu trọng số bằng định dạng 'checkpoint_path'
    craft.save_weights(checkpoint_path.format(epoch = 0))
    
    # Hàm callbacks
    callbacks = [modelckpt]

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = MyLRSchedule([50000, 200000], [FLAGS.init_learning_rate, FLAGS.init_learning_rate / 10. , FLAGS.init_learning_rate / 100.]))

    # Complie model
    print("[INFO] Biên dịch mô hình...")
    craft.compile(optimizer = optimizer, run_eagerly = True)

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