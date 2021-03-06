from lib import *
from file_utils import *

def getDetBoxes_core(image_path, text_map, link_map, text_threshold, link_threshold, low_text, s = True):
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    
    # prepare data
    link_map = link_map.copy()
    text_map = text_map.copy()
    img_h, img_w = text_map.shape

    """ labeling method """
    ret, text_score = cv2.threshold(text_map, low_text, 1, 0)
    ret, link_score = cv2.threshold(link_map, link_threshold, 1, 0)
    ret, text_score1 = cv2.threshold(text_map, low_text, 255, 0)
    ret, link_score1 = cv2.threshold(link_map, link_threshold, 255, 0)
    cv2.imwrite('result/' + filename + '_bi_text_map.jpg', text_score1)
    cv2.imwrite('result/' + filename + '_bi_link_score.jpg', link_score1)

    if s:
        text_score_comb = np.clip(text_score + link_score, 0, 1)
    else:
        text_score_comb = np.clip(text_score, 0, 1)
        
    label_n, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8),
                                                                        connectivity = 4)

    det = []
    mapper = []
    for k in range(1, label_n):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(text_map[labels == k]) < text_threshold:
            continue

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h

        tmp_text_map = text_map[sy:ey, sx:ex]
        tmp_labels = labels[sy:ey, sx:ex]

        tmp_link_score = link_score[sy:ey, sx:ex]
        tmp_text_score = text_score[sy:ey, sx:ex]

        # make segmentation map
        segmap = np.zeros(tmp_text_map.shape, dtype=np.uint8)
        segmap[tmp_labels == k] = 255
        segmap[np.logical_and(tmp_link_score == 1, tmp_text_score == 0)] = 0  # remove link area

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap = cv2.dilate(segmap, kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        start_idx = box.sum(axis = 1).argmin()
        box = np.roll(box, 4 - start_idx, 0)
        box = np.array(box)
        box += (sx, sy)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper

def getDetBoxes(image_path, textmap, linkmap, text_threshold, link_threshold, low_text, s = True):
    boxes, labels, mapper = getDetBoxes_core(image_path, textmap, linkmap, text_threshold, link_threshold, low_text, s)

    return boxes


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys

def get_result_img(image_path, image, score_text, score_link, text_threshold = 0.68, link_threshold = 0.4, low_text = 0.08, ratio_w = 1.0, ratio_h = 1.0, dirname = 'text_image'):
    boxes = getDetBoxes(image_path, score_text, score_link, text_threshold, link_threshold, low_text, s = False)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    saveResult(image_path, image, boxes, dirname = dirname, s = False)
    boxes = getDetBoxes(image_path, score_text, score_link, text_threshold, link_threshold, low_text, s = True)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    saveResult(image_path, image, boxes, dirname = dirname, s =True)