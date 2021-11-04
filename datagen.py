from lib import *
from text_utils import get_result_img
from augment import *

def normalizeMeanVariance(in_img, mean = (0.485, 0.456, 0.406), variance = (0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    
    return img

def four_point_transform(image, pts):
    max_x, max_y = np.max(pts[:, 0]).astype(np.int32), np.max(pts[:, 1]).astype(np.int32)

    dst = np.array([
        [0, 0],
        [image.shape[1] - 1, 0],
        [image.shape[1] - 1, image.shape[0] - 1],
        [0, image.shape[0] - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(dst, pts)
    warped = cv2.warpPerspective(image, M, (max_x, max_y))
    
    return warped

def gen_gaussian():
    sigma = 10
    spread = 3
    extent = int(spread * sigma)
    gaussian_heatmap = np.zeros([2 * extent, 2 * extent], dtype = np.float32)

    for i in range(2 * extent):
        for j in range(2 * extent):
            gaussian_heatmap[i, j] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                -1 / 2 * ((i - spread * sigma - 0.5) ** 2 + (j - spread * sigma - 0.5) ** 2) / (sigma ** 2))

    gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255).astype(np.uint8)

    # threshhold_guassian = cv2.applyColorMap(gaussian_heatmap, cv2.COLORMAP_JET)
    # cv2.imwrite(os.path.join("test_gaussian", 'test_gaussian.jpg'), threshhold_guassian)
    
    return gaussian_heatmap

def add_character(image, bbox):
    '''Phép biến đổi phối cảnh để có được bản đồ nhiệt của một ký tự'''
    top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
    if np.any(bbox < 0) or np.any(bbox[:, 0] > image.shape[1]) or np.any(bbox[:, 1] > image.shape[0]):	
        return image
    bbox -= top_left[None, :]
    transformed = four_point_transform(gen_gaussian().copy(), bbox.astype(np.float32))

    start_row = max(top_left[1], 0) - top_left[1]
    start_col = max(top_left[0], 0) - top_left[0]
    end_row = min(top_left[1] + transformed.shape[0], image.shape[0])
    end_col = min(top_left[0] + transformed.shape[1], image.shape[1])

    image[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += transformed[start_row:end_row - top_left[1],
                                                                       start_col:end_col - top_left[0]]

    return image

def generate_target(image_size, character_bbox):
    '''Bản đồ nhiệt đặc tính tạo ra các ký tự của toàn bộ bức tranh'''
    character_bbox = character_bbox.transpose(2, 1, 0)

    height, width, channel = image_size

    target = np.zeros([height, width], dtype=np.float32)

    for i in range(character_bbox.shape[0]):
        target = add_character(target, character_bbox[i])

    return target / 255.0, np.float32(target != 0)

def add_affinity(image, bbox_1, bbox_2):
    '''Lấy aff_heatmat của hai ký tự liền kề'''

    center_1, center_2 = np.mean(bbox_1, axis = 0), np.mean(bbox_2, axis = 0)
    tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
    bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
    tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
    br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

    affinity = np.array([tl, tr, br, bl])

    return add_character(image, affinity)

def generate_affinity(image_size, character_bbox, text):

    """
    Tạo bản đồ nhiệt toàn bộ ảnh aff
    :param image_size: shape = [3, h, w]
    :param character_bbox: [2, 4, num_characters]
    :param text: [num_words]
    :return:
    """

    character_bbox = character_bbox.transpose(2, 1, 0)

    height, width, channel = image_size

    target = np.zeros([height, width], dtype=np.float32)

    total_letters = 0

    for word in text:
        for char_num in range(len(word) - 1):
            target = add_affinity(target, character_bbox[total_letters].copy(),
                                   character_bbox[total_letters + 1].copy())
            total_letters += 1
        total_letters += 1

    return target / 255.0, np.float32(target != 0)

def procces_function(image, bbox, labels_text):
    image_shape = [image.shape[0], image.shape[1], image.shape[2]]
    weight, target = generate_target(image_shape, bbox.copy())
    weight_aff, target_aff = generate_affinity(image_shape, bbox.copy(), labels_text)
    
    return image, weight, target, weight_aff, target_aff


def count_samples(FLAGS):
    return len(glob(os.path.join(FLAGS.training_data_path, "*", "*.jpg")))

def generator(FLAGS):
    mat = scio.loadmat(os.path.join(FLAGS.training_data_path, 'gt.mat'))
    print('load gt.mat')
    # get information
    imnames = mat['imnames'][0]
    txt = mat['txt'][0]
    for no, i in enumerate(txt):
        all_words = []
        for j in i:
            all_words += [k for k in ' '.join(j.split('\n')).split() if k != '']
        txt[no] = all_words
    charBB = mat['charBB'][0]
    epoch = 1
    num = len(imnames)
    index = [v for v in range(num)]
    while True:
        random.shuffle(index)
        batch_image = []
        batch_label = []
        for i in index:
            try:
                image = plt.imread(os.path.join(FLAGS.training_data_path, imnames[i][0]))
                tmp = image.copy()
                bbox = charBB[i]
                text = txt[i]
                _, weight, target, weight_aff, target_aff = procces_function(tmp, bbox, text)
                label = np.dstack((weight, weight_aff))
                res_img, res_label = rand_augment(tmp, label)
                res_img = cv2.resize(res_img, dsize = (FLAGS.input_size, FLAGS.input_size), interpolation = cv2.INTER_LINEAR)
                #res_img = normalizeMeanVariance(res_img) //replace by preprocessing function
                res_label = cv2.resize(res_label, (FLAGS.input_size // 2, FLAGS.input_size // 2), interpolation = cv2.INTER_NEAREST)
                batch_image.append(res_img)
                batch_label.append(res_label)
                
                if len(batch_image) == FLAGS.batch_size:
                    yield (np.array(batch_image), np.array(batch_label))
                    batch_image = []
                    batch_label = []
            except Exception as e:
                import traceback
                if not FLAGS.suppress_warnings_and_error_messages:
                    traceback.print_exc()
                continue
        epoch += 1

# if __name__ == "__main__":
#     heatmap = gen_gaussian()
#     cv2.imwrite('heat.jpg', heatmap)