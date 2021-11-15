from enum import Flag
from lib import *
from net import CRAFT_model
from file_utils import list_files
from text_utils import get_result_img
from datagen import normalizeMeanVariance


parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type = int, default = 512) # kích thước đầu vào để đào tạo mạng
parser.add_argument('--model_name', type = str, default = 'resnet50')  # chọn model train
parser.add_argument('--gpu_list', type = str, default = '0', help = 'danh sách gpu sử dụng')
parser.add_argument('--text_threshold', default = 0.68, type = float, help = 'ngưỡng tin cậy văn bản')
parser.add_argument('--low_text', default = 0.4, type = float, help = 'văn bản điểm giới hạn thấp')
parser.add_argument('--link_threshold', default = 0.4, type = float, help = 'ngưỡng tin cậy liên kết')
parser.add_argument('--ratio_w', default = 1.0, type = int, help = 'tỷ lệ chiều rộng')
parser.add_argument('--ratio_h', default = 1.0, type = float, help = 'tỷ lệ chiều cao')
parser.add_argument('--checkpoint_path', type = str, default = 'tmp/checkpoint') # # đường dẫn đến một thư mục để lưu các điểm kiểm tra của mô hình trong quá trình đào tạo
parser.add_argument('--show_time', default = True, action = 'store_true', help = 'hiển thị thời gian xử lý')
parser.add_argument('--results_img', default = r'result_image/', type = str, help = 'Đường dẫn test')
parser.add_argument('--results_weights', default = r'results_weights', type = str, help = 'Đường dẫn trọng số region vs aff')
parser.add_argument('--test_folder', default = r'images', type = str, help = 'Đường dẫn test')

FLAGS = parser.parse_args()

result_folder = 'result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def load_image(img_path):
    """
    Load an image from file.
    :param img_path: Image file path, e.g. ``test.jpg`` or URL.
    :return: An RGB-image MxNx3.
    """
    img = io.imread(img_path)
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img

def predict(model, image_path, img, text_threshold = 0.68, link_threshold = 0.4, low_text = 0.08, ratio_w = 1.0, ratio_h = 1.0, dirname = 'result_image'):
    t0 = time.time()
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))
    # resize
    img = cv2.resize(img, (512, 512))
    src_img = normalizeMeanVariance(img)
    # src_img= np.expand_dims(src_img, axis=0)
    src_img = np.reshape(src_img, (1, 512, 512, 3))

    # lập bản đồ điểm và liên kết
    score_pre = model.predict(src_img)
    t0 = time.time() - t0

    t1 = time.time()

    # Xử lý
    score_pre = np.reshape(score_pre, (256, 256, 2))
    get_result_img(image_path, image, score_pre[:,:,0], score_pre[:,:,1], text_threshold, link_threshold, low_text, ratio_w, ratio_h, dirname)
    
    t1 = time.time() - t1

    if FLAGS.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    score_pre = cv2.resize(score_pre, (512, 512))
    score_txt = score_pre[:,:,0]
    score_link = score_pre[:,:,1]
    
    return score_txt, score_link

def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    # kiểm tra có thư mục results_img
    if not os.path.isdir(FLAGS.results_img):
        os.mkdir(FLAGS.results_img)

    # kiểm tra có thư mục results
    if not os.path.isdir(FLAGS.results_weights):
        os.mkdir(FLAGS.results_weights)

    """ Load model """
    # tạo đường dẫn lưu file
    checkpoint_path = os.path.sep.join([FLAGS.checkpoint_path, "model_craft_%s-{epoch:04d}.ckpt"%(FLAGS.model_name)])
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # Tạo một phiên bản mô hình mới
    new_model = CRAFT_model(FLAGS.model_name)

    # Tải trọng lượng đã lưu trước đó
    new_model.load_weights(latest).expect_partial()

    """ For test images in a folder """
    image_list, _, _ = list_files(FLAGS.test_folder)

    t = time.time()

    """ Test images """
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        image = Image.imread(image_path)
        start_time = time.time()
        score_txt, score_link = predict(new_model, image_path, image, FLAGS.text_threshold, FLAGS.link_threshold, FLAGS.low_text, FLAGS.ratio_w, FLAGS.ratio_h, FLAGS.results_img)
        print(time.time() * 1000 - start_time * 1000)

        plt.imsave(os.path.join(FLAGS.results_weights, '%s_region.jpg'%filename), score_txt)
        plt.imsave(os.path.join(FLAGS.results_weights, '%s_aff.jpg'%filename), score_link)

    print("Thời gian chạy : {}s".format(time.time() - t))

if __name__ == '__main__':
    test()