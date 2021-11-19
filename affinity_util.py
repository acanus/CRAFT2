# Usage : cal_affinity_boxes(region_boxes)
# Input : A list of region boxes for a single word
# return : A list of affinity boxes for a single word
from lib import *

epsilon = 1e-5

# Input : Two points
# Returns the distance between two points
def dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Input : Two points
# Returns the slope of a line determined by two points p1 and p2
def slope(p1,p2):
    return (p2[1] - p1[1])/(p2[0] - p1[0] + epsilon)

# Returns if the point p lies above the line given by st and slope
def above_line(p, st, slope):
    y = (p[0] - st[0]) * slope + st[1]
    return p[1] < y

# Reorder points in the clockwise direction from top left i.e. (tl, tr, br, bl)
# Input : A list of 4 points
# return : A list of 4 points
def reorder_points(points):
    
    # Point with minimum x
    ordered_points = sorted(points, key=lambda x: (x[0], x[1]))
    p1 = ordered_points[0]
    
    # Find 3rd point with middle slop
    slopes = [[slope(p1, p), p] for p in ordered_points[1:]]
    ordered_slopes = sorted(slopes, key=lambda x: x[0])
    slope_13, p3 = ordered_slopes[1]
    
    # Find 2nd point above 1-3 diagonal - guarantees clockwise now
    if above_line(ordered_slopes[0][1], p3, slope_13): 
        p2 = ordered_slopes[0][1]
        p4 = ordered_slopes[2][1]
        reverse = False
    else:
        p2 = ordered_slopes[2][1]
        p4 = ordered_slopes[0][1]
        reverse = True

    # Find the top left point.
    slope_24 = slope(p2, p4)
    if slope_13 < slope_24:
        if reverse:
            reorder_points = [p4, p1, p2, p3]
        else:
            reorder_points = [p2, p3, p4, p1]
    else:
        reorder_points = [p1, p2, p3, p4]

    return reorder_points

# Returns min distance between two boxes
# Input : Two Boxes
def min_box_dist(box1, box2):
    box_dist = [dist(p1, p2) for p1 in box1 for p2 in box2]
    return np.min(box_dist)

# Reorder character bounding boxes to get adjacent boxes
# Input : List of boxes
def reorder_box(boxes):
    
    # Compute distance matrix between every pair of boxes
    n = len(boxes)
    M = np.zeros((n,n), dtype = np.float32)
    for i in range(n):
        box1 = boxes[i]
        for j in range(i + 1, n):
            box2 = boxes[j]
            dist = min_box_dist(box1, box2)
            M[i][j] = M[j][i] = dist
    
    # Get boxes at extreme ends
    end_ind = np.argmax(M)
    inf = M[end_ind // n, end_ind % n] + 1 # Index from flattened matrix index
    for i in range(n):
        M[i, i] = inf
    
    cur_ind = end_ind // n
    reordered_boxes = [boxes[cur_ind]]
    for i in range(n - 1):
        M[:, cur_ind] = inf # To not repeat box
        closest_box = np.argmin(M[cur_ind])
        reordered_boxes.append(boxes[closest_box])
        cur_ind = closest_box
        
    return reordered_boxes

# Returns the area of the traingle formed by three points
# Input : Three points
def tri_area(p1, p2, p3):
    return abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) / 2

# Returns the area of the quadrilateral
# Input : Four points
def quad_area(p1, p2, p3, p4):
    p1, p2, p3, p4 = reorder_points([p1, p2, p3, p4])
    s1 = tri_area(p1,p2,p3)
    s2 = tri_area(p1,p3,p4)
    s3 = tri_area(p1,p2,p4)
    s4 = tri_area(p2,p3,p4)
    
    #Check if an actual convex rectangle is being formed 
    if s1 + s2 == s3 + s4:
        return s1 + s2
    else:
        return 0

# Calculate the center of quadrilateral - intersection of diagonals
# Input : A list of 4 points
def intersect(points):
    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = points
    x = ((x3 - x1) * (x4 - x2) * (y2 - y1) + x1 * (y3 - y1) * (x4 - x2) - x2 * (y4 - y2) * (x3 - x1)) \
        / ((y3 - y1) * (x4 - x2) - (y4 - y2) * (x3 - x1) + epsilon)
    y = (y3 - y1) * ((x4 - x2) * (y2 - y1) + (x1 - x2) * (y4 - y2)) \
        / ((y3 - y1) * (x4 - x2) - (y4 - y2) * (x3 - x1) + epsilon) + y1
    return [x, y]

# Returns the centroid of the triangle
# Input : Three points
def center(p1, p2, p3):
    points = np.array([p1, p2, p3])
    return [round(np.average(points[:, 0])), round(np.average(points[:, 1]))]

# Calculate possible vertices of affinity box
# A lits of 4 points
def cal_ppairs(points):
    box_c = intersect(points)
    p1, p2, p3, p4 = points
    return [[center(p1, p2, box_c), center(p3, p4, box_c)], [center(p1, p3, box_c), center(p2, p4, box_c)]]

def cal_abox(pairs1, pairs2):
    areas = [quad_area(p1,p2,p3,p4) for p1, p2 in pairs1 for p3, p4 in pairs2]
    # Check which pairs of points gives an actual convex quadrilateral
    max_ind = np.argmax(np.array(areas))
    
    abox = [pairs1[max_ind // 2][0], pairs1[max_ind // 2][1], pairs2[max_ind % 2][0], pairs2[max_ind % 2][1]]
    abox = reorder_points(abox)
    return abox

# Returns affinity_boxes
# Input : A list of char boxes for a single word
def cal_affinity_boxes(region_boxes):
    
    # Reorder boxes to get pairs of adjacent boxes
    region_boxes = reorder_box([reorder_points(r_box) for r_box in region_boxes])
    
    # Get point pairs in each character level region
    point_pairs = [cal_ppairs(r_box) for r_box in region_boxes] 
    affinity_boxes = []
    
    # Get affinity box for each pair of adjacent point pairs
    for i in range(len(point_pairs) - 1):
        a_box = reorder_points(cal_abox(point_pairs[i], point_pairs[i + 1]))
        affinity_boxes.append(a_box)
    return affinity_boxes

# if __name__ == '__main__':
#     # print(reorder_points([[251, 96], [284, 112], [267, 112], [253, 118]]))
#     # print(reorder_points([[0, 0], [4, 4], [4, 0], [0, 4]]))
#     # print(reorder_points([[56, 25], [85, 45], [25, 80], [15, 45]]))

#     # print(reorder_box([[[0, 0], [4, 4], [4, 0], [0, 4]],
#     #                    [[12, 0], [16, 4], [16, 0], [12, 4]],
#     #                    [[16, 0], [20, 4], [20, 0], [16, 4]],
#     #                    [[4, 0], [8, 4], [8, 0], [4, 4]],
#     #                    [[8, 0], [12, 4], [12, 0], [8, 4]]]))

#     print(cal_affinity_boxes([[[0, 0], [4, 4], [4, 0], [0, 4]],
#                               [[12, 0], [16, 4], [16, 0], [12, 4]],
#                               [[16, 0], [20, 4], [20, 0], [16, 4]],
#                               [[4, 0], [8, 4], [8, 0], [4, 4]],
#                               [[8, 0], [12, 4], [12, 0], [8, 4]]]))