import cv2
import numpy as np
import os

# NMS 方法（Non Maximum Suppression，非极大值抑制）
def nms(boxes, overlapThresh = 0.3):
    if len(boxes) == 0:
        return []
 
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    pick = []
 
    # 取四个坐标数组
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
 
    # 计算面积数组
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
 
    # 按得分排序（如没有置信度得分，可按坐标从小到大排序，如右下角坐标）
    idxs = np.argsort(y2)
 
    # 开始遍历，并删除重复的框
    while len(idxs) > 0:
        # 将最右下方的框放入pick数组
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # 找剩下的其余框中最大坐标和最小坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # 计算重叠面积占对应框的比例，即 IoU
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
 
        # 如果 IoU 大于指定阈值，则删除
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
 
    return boxes[pick].astype("int")

def dist(x1, y1, x2, y2):
    '''
    计算两点之间的距离
    '''
    distance = np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
    return distance

def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    """
    计算两个矩形框的距离
    input：两个矩形框，分别左上角和右下角坐标
    return：像素距离
    """
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0

def two2one(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    """
    将两个矩形框，变成一个更大的矩形框
    input：两个矩形框，分别左上角和右下角坐标
    return：融合后矩形框左上角和右下角坐标
    """
    x = min(x1, x2)
    y = min(y1, y2)
    xb = max(x1b, x2b)
    yb = max(y1b, y2b)
    return x, y, xb, yb

def box_merge(boxes):
    """
    多box，最终融合距离近的，留下新的，或未被融合的
    input：多box的列表，例如：[[12,23,45,56],[36,25,45,63],[30,25,60,35]]
    return：新的boxes，这里面返回的结果是这样的，被合并的box会置为[]，最终返回的，可能是这样[[],[],[50,23,65,50]]
    """
    keep = []
    ## fisrt boxes add, boxes minus
    if len(boxes) > 0:
        for bi in range(len(boxes)):
            for bj in range(len(boxes)):
                if bi != bj:
                    if len(boxes[bi]) == 4 and len(boxes[bj]) == 4:
                        x1, y1, x1b, y1b = int(boxes[bi][0]), int(boxes[bi][1]), int(boxes[bi][2]), int(boxes[bi][3])
                        x2, y2, x2b, y2b = int(boxes[bj][0]), int(boxes[bj][1]), int(boxes[bj][2]), int(boxes[bj][3])
                        dis = rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b)
                        if dis < 30:
                            boxes[bj][0], boxes[bj][1], boxes[bj][2], boxes[bj][3] = two2one(x1, y1, x1b, y1b, x2, y2, x2b, y2b)
                            boxes[bi] = [0,0,0,0]
    return boxes

def find_potential_hanzi_boxes(boxes, lower_bound = 60, upper_bound = 190):
    """
    找到汉字的box
    input：boxes，多个box的列表，例如：[[12,23,45,56],[36,25,45,63],[30,25,60,35]]
    return：汉字的box，例如：[12,23,45,56]
    """
    hanzi_box = []
    for box in boxes:
        # print(box[2] -box[0],box[3] -box[1])
        if len(box) == 4 and (upper_bound > box[2] -box[0] > lower_bound and upper_bound > box[3] - box[1] > lower_bound) or (upper_bound/3 > box[2] -box[0] > lower_bound/3 and upper_bound/3 > box[3] - box[1] > lower_bound/3):
            # print(box[2] -box[0],box[3] -box[1])
            hanzi_box.append(box)
    return hanzi_box

def hanzi_box_detect(img, mode = 'red'):
    # 灰度化
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = img.shape[:2]
    gray = np.zeros((h,w), np.uint8)
    if mode == 'red':
        gray[:,:] = img[:,:,2]
    if mode == 'green':
        gray[:,:] = img[:,:,1]
    gray = np.clip(gray, 0, 205) # 压制高光
    cv2.imshow('gray', gray)

    vis = img.copy()
    orig = img.copy()
    # 调用 MSER 算法
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)  # 获取文本区域
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  # 绘制文本区域
    cv2.polylines(vis, hulls, 1, (0, 255, 0), 2)
    cv2.imshow('img', vis)
    
    # 将不规则检测框处理成矩形框
    keep = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        keep.append([x, y, x + w, y + h])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.imshow("hulls", vis)

    boxes = nms(np.array(keep), overlapThresh = 0.3)
    # boxes = box_merge(boxes) # 交叉矩形合并
    for (x, y, x2, y2) in boxes:
        cv2.rectangle(orig, (x, y), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("boxes", orig)

    hanzi_box = find_potential_hanzi_boxes(boxes, lower_bound = 60, upper_bound = 190)
    
    return hanzi_box

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, 255*s, 255*v

def in_hanzi_box(center, hanzi_box):
    # print(center)
    # print(hanzi_box)
    for box in hanzi_box:
        if center[0] > box[0] and center[0] < box[2] and center[1] > box[1] and center[1] < box[3]:
            return True
    return False

def color_detect_old(img):  # 弃用
    # 灰度化
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color = [0,0,0]
    centers = []
    h,w = img.shape[:2]
    gray = np.zeros((h,w), np.uint8)
    gray[:,:] = img[:,:,2]

    vis = img.copy()
    orig = img.copy()
    # 调用 MSER 算法
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)  # 获取文本区域
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  # 绘制文本区域
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    # cv2.imshow('img', vis)
    for region in regions:
        center = region.mean(axis=0).astype(int)
        centers.append(center)

    # 将不规则检测框处理成矩形框
    keep = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        keep.append([x, y, x + w, y + h])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)
    cv2.imshow("hulls", vis)

    boxes = nms(np.array(keep), overlapThresh = 0.3)
    # boxes = box_merge(boxes) # 交叉矩形合并
    for (x, y, x2, y2) in boxes:
        cv2.rectangle(orig, (x, y), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow("boxes", orig)

    hanzi_box = find_potential_hanzi_boxes(boxes, lower_bound = 60, upper_bound = 190)
    for (x, y, x2, y2) in hanzi_box:
        cv2.rectangle(orig, (x, y), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("hanzi_box", orig)

    color_num = 0
    color_center = [0,0]
    for center in centers:
        print("center:", center)
        if in_hanzi_box(center, hanzi_box) == False:
            continue
        if int(img[center[1],center[0],0]) + int(img[center[1],center[0],1]) + int(img[center[1],center[0],2]) > 255*2.5:
            continue
        cv2.circle(vis, (center[0], center[1]), 3, (0, 0, 0), -1)
        num = 0
        for c in centers:
            if dist(center[0], center[1], c[0], c[1]) > 0 and dist(center[0], center[1], c[0], c[1]) < 10:
                num += 1
        if num > color_num:
            color_num = num
            color_center = center

    cv2.circle(vis, (color_center[0], color_center[1]), 7, (0, 0, 255), -1)
    cv2.imshow("hulls", vis)

    # 计算邻域颜色均值
    sum = 0
    for i in range(color_center[0]-5, color_center[0]+5):
        for j in range(color_center[1]-5, color_center[1]+5):
            light = int(img[i,j,0]) + int(img[i,j,1]) + int(img[i,j,2])
            if light > 255:
                sum += 1
                color[0] += img[i,j,0]
                color[1] += img[i,j,1]
                color[2] += img[i,j,2]
    color[0] /= sum
    color[1] /= sum
    color[2] /= sum

    return color

def ensure_dir(dir_path):
    '''生成文件夹'''
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

# if __name__ == '__main__':
#     # 读取图片
#     imagePath = r"origin_img\red\zhadan\14.jpg"
#     img = cv2.imread(imagePath)
#     orig = img.copy()

#     hanzi_box = hanzi_box_detect(img)
#     print(hanzi_box)
#     for (x, y, x2, y2) in hanzi_box:
#         cv2.rectangle(orig, (x, y), (x2, y2), (0, 255, 0), 2)
#     cv2.imshow("hanzi_box", orig)

#     # color = color_detect(img)
#     # # print("color:", color)
#     # h,s,v = rgb2hsv(color[2], color[1], color[0])
#     # print("hsv:", h,s,v)

#     cv2.waitKey(0)

# if __name__ == '__main__':
#     qizi = ['dilei', 'gongbin', 'junqi', 'junzhang', 'lianzhang', 'lvzhang',
#             'paizhang', 'shizhang', 'siling', 'tuanzhang', 'yinzhang', 'zhadan']

#     print('extract red qizi')
#     for i in range(0, 12):
#         print(qizi[i])
#         save_dir = './extract_img_mser/' + qizi[i] # 保存文件夹
#         img_dir = './origin_img/red/' + qizi[i] # 来源文件夹

#         for _, _, files in os.walk(img_dir):
#             # 遍历文件
#             # print(files)
#             for f in files:
#                 img_file_dir = img_dir + '/' + f
#                 ensure_dir(save_dir)
#                 save_file_dir = save_dir + '/ex_red_' + f
#                 img = cv2.imread(img_file_dir)  # 读取图片
#                 img = img[0:719,0:700,:]
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 # cv2.imshow('img', img)
#                 hanzi_boxes = hanzi_box_detect(img)
#                 print(hanzi_boxes)
#                 i = 1
#                 for (x, y, x2, y2) in hanzi_boxes:
#                     red_Hanzi = gray[x:x2,y:y2]   
#                     if red_Hanzi is None or red_Hanzi.shape[0] == 0 or red_Hanzi.shape[1] == 0:
#                         print('Failed')
#                     else:
#                         print('Success')
#                         save_file_dir_tmp = save_file_dir[:-4] + '_' + str(i) + '.jpg'
#                         # print(save_file_dir_tmp)
#                         if x2-x > y2-y:
#                             red_Hanzi = cv2.resize(red_Hanzi, (150, 100))
#                         else:
#                             red_Hanzi = cv2.resize(red_Hanzi, (100, 150))
#                             red_Hanzi = np.rot90(red_Hanzi)
#                         cv2.imwrite(save_file_dir_tmp, red_Hanzi)
#                         i += 1
#     cv2.waitKey(0)


# if __name__ == '__main__':
#     qizi = ['dilei', 'gongbin', 'junqi', 'junzhang', 'lianzhang', 'lvzhang',
#             'paizhang', 'shizhang', 'siling', 'tuanzhang', 'yinzhang', 'zhadan']

#     print('extract red qizi')
#     for i in range(0, 12):
#         print(qizi[i])
#         save_dir = './extract_img_mser/' + qizi[i] # 保存文件夹
#         img_dir = './origin_img/green/' + qizi[i] # 来源文件夹

#         for _, _, files in os.walk(img_dir):
#             # 遍历文件
#             # print(files)
#             for f in files:
#                 img_file_dir = img_dir + '/' + f
#                 ensure_dir(save_dir)
#                 save_file_dir = save_dir + '/ex_red_' + f
#                 img = cv2.imread(img_file_dir)  # 读取图片
#                 img = img[0:719,600:1279,:]
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 # cv2.imshow('img', img)
#                 hanzi_boxes = hanzi_box_detect(img, mode='green')
#                 print(hanzi_boxes)
#                 i = 1
#                 for (x, y, x2, y2) in hanzi_boxes:
#                     red_Hanzi = gray[x:x2,y:y2]   
#                     if red_Hanzi is None or red_Hanzi.shape[0] == 0 or red_Hanzi.shape[1] == 0:
#                         print('Failed')
#                     else:
#                         print('Success')
#                         save_file_dir_tmp = save_file_dir[:-4] + '_' + str(i) + '.jpg'
#                         # print(save_file_dir_tmp)
#                         if x2-x > y2-y:
#                             red_Hanzi = cv2.resize(red_Hanzi, (150, 100))
#                         else:
#                             red_Hanzi = cv2.resize(red_Hanzi, (100, 150))
#                             red_Hanzi = np.rot90(red_Hanzi)
#                         cv2.imwrite(save_file_dir_tmp, red_Hanzi)
#                         i += 1
#     cv2.waitKey(0)

if __name__ == "__main__":
    hsv = rgb2hsv(158, 189, 119)
    print(hsv)