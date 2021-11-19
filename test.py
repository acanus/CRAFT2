from lib import *
from net import CRAFT_model
from file_utils import list_files, saveResult
from inference_util import getDetBoxes, adjustResultCoordinates
from img_util import load_image, img_resize, img_normalize, to_heat_map

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type = int, default = 600)  # kích thước hình ảnh để đào tạo
parser.add_argument('--model_name', type = str, default = "vgg16")  # chọn model train
parser.add_argument('--checkpoint_path', type = str, default = 'tmp') # # đường dẫn đến một thư mục để lưu các điểm kiểm tra của mô hình trong quá trình đào tạo
parser.add_argument('--gpu_list', type = str, default = '0', help = 'list of gpu to use')
parser.add_argument('--text_threshold', default = 0.7, type = float, help = 'text confidence threshold')
parser.add_argument('--low_text', default = 0.4, type = float, help = 'text low-bound score')
parser.add_argument('--link_threshold', default = 0.4, type = float, help = 'link confidence threshold')
parser.add_argument('--canvas_size', default = 1280, type = int, help = 'image size for inference')
parser.add_argument('--mag_ratio', default = 1., type = float, help = 'image magnification ratio')
parser.add_argument('--show_time', default = True, action = 'store_true', help='show processing time')
parser.add_argument('--s', default = True, action = 'store_true', help='hiển thị kết quả ký tự hay từ, True: từ, False: ký tự')
parser.add_argument('--test_folder', default = r'datasets\ICDAR_15\test',
                    type = str, help = 'folder path to input images')

FLAGS = parser.parse_args()

result_folder = 'results/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def predict(model, image, text_threshold, link_threshold, low_text, s):
    t0 = time.time()

    # resize
    h, w = image.shape[:2]
    mag_ratio = FLAGS.img_size / max(h, w)
    # img_resized, target_ratio = img_resize(image, FLAGS.mag_ratio, FLAGS.canvas_size, interpolation=cv2.INTER_LINEAR)
    img_resized, target_ratio = img_resize(image, mag_ratio, FLAGS.canvas_size, interpolation = cv2.INTER_LINEAR)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = img_normalize(img_resized)

    # make score and link map
    score_text, score_link = model.predict(np.array([x]))
    score_text = score_text[0]
    score_link = score_link[0]

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, s)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    white_img = np.ones((render_img.shape[0], 10, 3), dtype = np.uint8) * 255
    ret_score_text = np.hstack((to_heat_map(render_img), white_img, to_heat_map(score_link)))
    # ret_score_text = to_heat_map(render_img)

    if FLAGS.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, ret_score_text


def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    """ Load model """
    print("[INFO] tải trọng số mô hình...")
    model = CRAFT_model(FLAGS.model_name)
    # tạo đường dẫn lưu model
    checkpoint_path = os.path.sep.join([FLAGS.checkpoint_path, "model_craft_%s.ckpt"%(FLAGS.model_name)])
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest).expect_partial()

    """ For test images in a folder """
    image_list, _, _ = list_files(FLAGS.test_folder)

    t = time.time()

    """ Test images """
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        image = load_image(image_path)
        start_time = time.time()
        bboxes, score_text = predict(model, image, FLAGS.text_threshold, FLAGS.link_threshold, FLAGS.low_text, FLAGS.s)
        print(time.time() * 1000 - start_time * 1000)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        saveResult(image_path, image[:, :, ::-1], bboxes, dirname = result_folder)

    print("elapsed time : {}s".format(time.time() - t))


if __name__ == '__main__':
    test()