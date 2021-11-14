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
    mean = 0
    radius = 1.5
    # a = 1 / (2 * np.pi * (radius ** 2))
    a = 1.
    x0, x1 = np.meshgrid(np.arange(-3, 3, 0.01), np.arange(-3, 3, 0.01))
    x = np.append([x0.reshape(-1)], [x1.reshape(-1)], axis = 0).T

    m0 = (x[:, 0] - mean) ** 2
    m1 = (x[:, 1] - mean) ** 2
    gaussian_heatmap = a * np.exp(-0.5 * (m0 + m1) / (radius ** 2))
    gaussian_heatmap = gaussian_heatmap.reshape(len(x0), len(x1))
    gaussian_heatmap = (gaussian_heatmap / np.max(gaussian_heatmap) * 255.0).astype(np.uint8)
 
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

class SynthTextDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, input_size, batch_size = 32, shuffle = True, augmentation = False):
        self.augmentation = augmentation
        self.mat = scio.loadmat(os.path.join(data_dir, 'gt.mat'))
        self.imnames = self.mat['imnames'][0]
        self.txt = self.mat['txt'][0]
        for no, i in enumerate(self.txt):
            all_words = []
            for j in i:
                all_words += [k for k in ' '.join(j.split('\n')).split() if k != '']
            self.txt[no] = all_words
        self.charBB = self.mat['charBB'][0]
        self.input_size = input_size
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.imnames)
        self.indexes = np.arange(self.num_samples)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = []
        Y = []
        for i, index in enumerate(indexes):
            image = plt.imread(os.path.join(self.data_dir, self.imnames[index][0]))
            image1 = cv2.imread(os.path.join(self.data_dir, self.imnames[index][0]))
            cv2.imwrite("test_image_%s.jpg"%index, image1)
            tmp = image.copy()
            bbox = self.charBB[index]
            text = self.txt[index]
            _, weight, target, weight_aff, target_aff = procces_function(tmp, bbox, text)
            label = np.dstack((weight, weight_aff))
            if self.augmentation:
                res_img, res_label = rand_augment(tmp, label)
            else:
                res_img, res_label = tmp, label
            res_img = cv2.resize(res_img, dsize = (self.input_size[1], self.input_size[0]), interpolation = cv2.INTER_LINEAR)
            res_img = normalizeMeanVariance(res_img) # replace by preprocessing function
            res_label = cv2.resize(res_label, (self.input_size[1] // 2, self.input_size[0] // 2), interpolation = cv2.INTER_NEAREST)
            X.append(res_img)
            Y.append(res_label)
        return np.array(X), np.array(Y)


# if __name__ == "__main__":
#     heatmap = gen_gaussian()
#     cv2.imwrite('heat.jpg', heatmap)
