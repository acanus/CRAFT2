from lib import *
from gaussian import GaussianGenerator
from data_prepare import Datagenerator
from fake import Fake
from data_util import load_data
from net import CRAFT_model

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type = int, default = 600)  # kích thước hình ảnh để đào tạo
parser.add_argument('--init_learning_rate', type = float, default = 0.001)  # tỷ lệ học tập ban đầu
parser.add_argument('--lr_decay_rate', type = float, default = 0.94) # tỷ lệ phân rã cho tỷ lệ học tập
parser.add_argument('--lr_decay_steps', type = int, default = 65) # số bước mà sau đó tốc độ học được giảm dần theo tốc độ giảm dần
parser.add_argument('--batch_size', type = int, default = 4)  # kích thước lô để đào tạo
parser.add_argument('--max_epochs', type = int, default = 800)  # số kỷ nguyên tối đa
parser.add_argument('--gpu_list', type = str, default = '0')  # list of gpus to use
parser.add_argument('--use_fake', type = bool, default = True)  # list of gpus to use
parser.add_argument('--checkpoint_path', type = str, default = 'tmp') # # đường dẫn đến một thư mục để lưu các điểm kiểm tra của mô hình trong quá trình đào tạo
parser.add_argument('--model_name', type = str, default = "resnet50")  # chọn model train

# path to training data
parser.add_argument('--truth_data_path', type = str, default = 'datasets/synthtext/SynthText')
parser.add_argument('--pseudo_data_path', type = str, default = 'datasets/ICDAR_15')
parser.add_argument('--val_data_path', type = str, default = 'datasets/ICDAR_15')  # Đường dẫn dữ liệu đánh giá

# parser.add_argument('--pseudo_data_path', type = str, default = 'datasets\ICDAR_13')
# parser.add_argument('--val_data_path', type = str, default = 'datasets\ICDAR_13')  # Đường dẫn dữ liệu đánh giá
parser.add_argument('--max_image_size', type = int, default = 1280)
parser.add_argument('--vis', type = bool, default = True)
parser.add_argument('--vis_num_batch_size', type = bool, default = 5) # cứ sao 50 batch size sẽ hiển thị kết quả train 1 lần
parser.add_argument('--load_weight', type = bool, default = False)
FLAGS = parser.parse_args()

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, boundaries, learning_rate):
        self.boundaries = boundaries
        self.learning_rate = learning_rate

    def __call__(self, step):
        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.boundaries, self.learning_rate)
        return  learning_rate_fn(step)

# def lr_decay(epoch):
#     return FLAGS.init_learning_rate * np.power(FLAGS.lr_decay_rate, epoch // FLAGS.lr_decay_steps)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    # kiểm tra xem đường dẫn điểm kiểm tra có tồn tại không
    if not os.path.exists(FLAGS.checkpoint_path):
        os.mkdir(FLAGS.checkpoint_path)

    # Khởi tạo mạng nơ-ron
    print("[INFO] Khởi tạo mô hình...")
    craft = CRAFT_model(FLAGS.model_name, vis = FLAGS.vis, num_batch_size = FLAGS.vis_num_batch_size)

    # tạo đường dẫn lưu model
    checkpoint_path = os.path.sep.join([FLAGS.checkpoint_path, "model_craft_%s.ckpt"%(FLAGS.model_name)])
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # tải trọng số để train tiếp
    if FLAGS.load_weight:
        print("[INFO] tải trọng số mô hình...")
        craft.load_weights(latest)

    # Optimizer
    # opt = tf.keras.optimizers.Adam(FLAGS.init_learning_rate)
    opt = tf.keras.optimizers.Adam(learning_rate = MyLRSchedule([2500, 10000], [FLAGS.init_learning_rate, FLAGS.init_learning_rate / 10. , FLAGS.init_learning_rate / 100.]))

    # Complie model
    print("[INFO] Biên dịch mô hình...")
    craft.compile(optimizer = opt, run_eagerly = True)

    # GaussianGenerator
    gaus = GaussianGenerator()

    # Tải dữ liệu
    print("[INFO] Tải dữ liệu SynthText...")
    true_sample_list = load_data(os.path.join(FLAGS.truth_data_path, 'train_gt.pkl'))
    train_sample_list = true_sample_list
    np.random.shuffle(train_sample_list)

    if FLAGS.use_fake:
        print("[INFO] Tải dữ liệu ICDAR_15...")
        pseudo_sample_list = load_data(os.path.join(FLAGS.pseudo_data_path, 'train_gt.pkl'))
        np.random.shuffle(pseudo_sample_list)
        train_generator = Datagenerator(craft, gaus, [train_sample_list, pseudo_sample_list], [5, 1], [False, True], FLAGS.img_size, FLAGS.batch_size)
    else:
        train_generator = Datagenerator(craft, gaus, [train_sample_list], [1], [False], FLAGS.img_size, FLAGS.batch_size)
    
    # tạo kiểm soát mô hình
    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decay)
    modelckpt = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_freq = 50 * FLAGS.batch_size,  save_weights_only = True, verbose = 1)

    # steps per epoch
    steps_per_epoch = len(train_generator)

    print("[INFO] Huấn luyện mạng...")
    H = craft.fit(train_generator,
                steps_per_epoch = steps_per_epoch,
                initial_epoch = 0,
                epochs = FLAGS.max_epochs,
                callbacks = [modelckpt],
                )

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