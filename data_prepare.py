from tensorflow.python.keras.backend import print_tensor
from lib import *
from affinity_util import reorder_points
from img_util import load_sample, img_normalize, load_image
from pseudo_util import watershed, crop_image, find_char_box, divide_region, un_warping, conf, enlarge_char_boxes


class Datagenerator(tf.keras.utils.Sequence):
    def __init__(self, base_model, gaus, train_sample_lists, train_sample_probs, fakes, img_size, batch_size, is_train = True):
        super().__init__()
        assert len(train_sample_lists) == len(train_sample_probs)
        assert len(train_sample_lists) == len(fakes)
        self.base_model = base_model
        self.gaus = gaus
        self.train_sample_lists = train_sample_lists
        self.fakes = fakes
        self.train_sample_probs = np.array(train_sample_probs) / np.sum(train_sample_probs)
        self.sample_count_list = [len(sample_list) for sample_list in train_sample_lists]
        self.sample_idx_list = [0] * len(train_sample_lists)
        self.sample_mark_list = list(range(len(train_sample_lists)))
        self.img_size = img_size
        self.batch_size = batch_size
        self.is_train = is_train

    def __len__(self):
        return math.ceil(np.sum(self.sample_count_list) / self.batch_size)

    def __getitem__(self, index):
        
        images = list()
        word_boxes_list = list()
        word_lengths_list = list()
        region_scores = list()
        affinity_scores = list()
        confidence_score_list = list()
        fg_masks = list()
        bg_masks = list()
        word_count_list = list()

        for i in range(self.batch_size):
            if self.is_train:
                sample_mark = np.random.choice(self.sample_mark_list, p = self.train_sample_probs)

                if sample_mark == self.sample_mark_list[0]:
                    new_sample_list = list()

                    if len(self.train_sample_lists[sample_mark][self.sample_idx_list[sample_mark]]) == 5:
                        img_path, word_boxes, words, char_boxes_list, _ = self.train_sample_lists[sample_mark][self.sample_idx_list[sample_mark]]
                    else:
                        img_path, word_boxes, words, char_boxes_list = self.train_sample_lists[sample_mark][self.sample_idx_list[sample_mark]]
                    confidence_list = [1] * len(word_boxes)
                    
                elif sample_mark == self.sample_mark_list[1]:
                    if self.fakes[sample_mark]:

                        new_sample_list = list()

                        if len(self.train_sample_lists[sample_mark][self.sample_idx_list[sample_mark]]) == 5:
                            img_path, word_boxes, words, char_boxes_list, _ = self.train_sample_lists[sample_mark][self.sample_idx_list[sample_mark]]
                        else:
                            img_path, word_boxes, words, char_boxes_list = self.train_sample_lists[sample_mark][self.sample_idx_list[sample_mark]]
                        
                        img = load_image(img_path)
                        char_boxes_list = list()

                        confidence_list = list()

                        for word_box, word in zip(word_boxes, words):
                            char_boxes, confidence = self.fake_char_boxes(img, word_box, len(word))
                            char_boxes_list.append(char_boxes)
                            confidence_list.append(confidence)
            else:
                sample_mark = np.random.choice(self.sample_mark_list, p = self.train_sample_probs)
                if self.fakes[sample_mark]:
                    break

            self.sample_idx_list[sample_mark] += 1
            if self.sample_idx_list[sample_mark] >= self.sample_count_list[sample_mark]:
                self.sample_idx_list[sample_mark] = 0
                np.random.shuffle(self.train_sample_lists[sample_mark])

            img, word_boxes, char_boxes_list, region_box_list, affinity_box_list, img_shape = load_sample(img_path, self.img_size, word_boxes, char_boxes_list)

            images.append(img)
            word_count = min(len(word_boxes), len(words))
            word_boxes = np.array(word_boxes[:word_count], dtype = np.int32) // 2
            word_boxes_list.append(word_boxes)
            word_count_list.append(word_count)

            word_lengths = [len(words[j]) if len(char_boxes_list[j]) == 0 else 0 for j in range(word_count)]
            word_lengths_list.append(word_lengths)

            height, width = img.shape[:2]
            heat_map_size = (height // 2, width // 2)

            mask_shape = (img_shape[1] // 2, img_shape[0] // 2)
            confidence_score = np.ones(heat_map_size, dtype = np.float32)
            for word_box, confidence_value in zip(word_boxes, confidence_list):
                if confidence_value == 1:
                    continue
                tmp_confidence_score = np.zeros(heat_map_size, dtype = np.uint8)
                cv2.fillPoly(tmp_confidence_score, [np.array(word_box)], 1)
                tmp_confidence_score = np.float32(tmp_confidence_score) * confidence_value
                confidence_score = np.where(tmp_confidence_score > confidence_score, tmp_confidence_score, confidence_score)
            confidence_score_list.append(confidence_score)

            fg_mask = np.zeros(heat_map_size, dtype = np.uint8)
            cv2.fillPoly(fg_mask, [np.array(word_box) for word_box in word_boxes], 1)
            fg_masks.append(fg_mask)
            bg_mask = np.zeros(heat_map_size, dtype = np.float32)
            bg_mask[:mask_shape[0], :mask_shape[1]] = 1
            bg_mask = bg_mask - fg_mask
            bg_mask = np.clip(bg_mask, 0, 1)
            bg_masks.append(bg_mask)

            region_score = self.gaus.gen(heat_map_size, np.array(region_box_list) // 2)
            region_scores.append(region_score)

            affinity_score = self.gaus.gen(heat_map_size, np.array(affinity_box_list) // 2)
            affinity_scores.append(affinity_score)

        max_word_count = np.max(word_count_list)
        max_word_count = max(1, max_word_count)
        new_word_boxes_list = np.zeros((self.batch_size, max_word_count, 4, 2), dtype = np.int32)
        new_word_lengths_list = np.zeros((self.batch_size, max_word_count), dtype = np.int32)
        for i in range(self.batch_size):
            if word_count_list[i] > 0:
                new_word_boxes_list[i, :word_count_list[i]] = np.array(word_boxes_list[i])
                new_word_lengths_list[i, :word_count_list[i]] = np.array(word_lengths_list[i])

        images = np.array(images)
        region_scores = np.array(region_scores, dtype = np.float32)
        affinity_scores = np.array(affinity_scores, dtype = np.float32)
        confidence_scores = np.array(confidence_score_list, dtype = np.float32)
        fg_masks = np.array(fg_masks, dtype = np.float32)
        bg_masks = np.array(bg_masks, dtype = np.float32)

        return [images], [region_scores, affinity_scores, confidence_scores, fg_masks, bg_masks]

    def fake_char_boxes(self, src, word_box, word_length):
        img, src_points, crop_points = crop_image(src, word_box, dst_height = 64.)
        h, w = img.shape[:2]
        if min(h, w) == 0:
            confidence = 0.5
            region_boxes = divide_region(word_box, word_length)
            region_boxes = [reorder_points(region_box) for region_box in region_boxes]
            return region_boxes, confidence
        img = img_normalize(img)
        # print(img.shape)

        region_score, _ = self.base_model.predict(np.array([img]))
        heat_map = region_score[0] * 255.
        heat_map = heat_map.astype(np.uint8)
        marker_map = watershed(heat_map)
        region_boxes = find_char_box(marker_map)
        confidence = conf(region_boxes, word_length)
        if confidence <= 0.5:
            confidence = 0.5
            region_boxes = divide_region(word_box, word_length)
            region_boxes = [reorder_points(region_box) for region_box in region_boxes]
        else:
            region_boxes = np.array(region_boxes) * 2
            region_boxes = enlarge_char_boxes(region_boxes, crop_points)
            region_boxes = [un_warping(region_box, src_points, crop_points) for region_box in region_boxes]
            # print(word_box, region_boxes)

        return region_boxes, confidence

