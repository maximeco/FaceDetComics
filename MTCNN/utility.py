import os
import numpy as np
import copy
import cv2 as cv
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Utils 
def load_images_from_folder(folder):
    '''
    Load all images from a given folder and return them in an array

    :param str folder: The path to the needed folder
    :return: Array of images
    :return: Array of paths
    '''
    images = []
    paths = []
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filepath = subdir + os.sep + filename
            
            img = cv.imread(filepath)
            # Make sure to deleter any "checkpoints" created by jupyter
            if img is not None and "checkpoints" not in filepath:
                images.append(img)
                paths.append(filepath)
    return images, paths


def load_all_groundtruth(folder, shape):
    '''
    Load all groundtruths from a given folder and return them in an array.
    Return all of the groundtruth ("faces, pannel, characters etc...")

    :param str folder: The path to the needed folder
    :param int shape:  Usefull when groundtruth has already been manipulated, default = 5 
    (class, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    :return: Array of groundtruths
    :return: Array of paths
    '''
    
    groundtruth = []
    path = []
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filepath = subdir + os.sep + filename
                      
            f = open(filepath, "r")
            if "checkpoints" not in filepath:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    groundtruth.append(np.loadtxt(f, dtype = int))
                    path.append(filepath)
    
    groundtruth = np.array(groundtruth)
    groundtruth = [x.reshape(-1, shape) for x in groundtruth]
    return groundtruth, path

def load_face_groundtruth_from_folder(folder, shape = 5):
    '''
    Load all groundtruths from a given folder and return them in an array.
    Return only the faces gt

    :param str folder: The path to the needed folder
    :param int shape:  Usefull when groundtruth has already been manipulated, default = 5 
    (class, top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    :return: Array of groundtruths
    :return: Array of paths
    '''
    
    groundtruth = []
    path = []
    for subdir, dirs, files in os.walk(folder):
        for filename in files:
            filepath = subdir + os.sep + filename
                      
            f = open(filepath, "r")
            if "checkpoints" not in filepath:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    groundtruth.append(np.loadtxt(f, dtype = int))
                    path.append(filepath)
    
    groundtruth = np.array(groundtruth)
    groundtruth = [x.reshape(-1, shape) for x in groundtruth]
    return  [gt[((gt[:, 0] == 7) | (gt[:, 0] == 11))] for gt in groundtruth], path

def resize_img(image, width=None, height=None, inter=cv.INTER_AREA):
    '''
    Resize an images to a given width or height while conserving the ratio aspect

    :param image: given image
    :param int width: desired final width 
    :param int width: desired final height
    :Note : if both are given, height will be ignored to conserve aspect ratio
    :return: Array of groundtruths
    :return: Array of paths
    '''
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

def resize_img_and_gt(width, images, groundtruth):
    '''
    Resize an image to a given width or height while conserving the ratio aspect

    :param image: given image
    :param int width: desired final width 
    :param int width: desired final height
    :Note : if both are given, height will be ignored to conserve aspect ratio
    :return: copy of the image resized
    ''' 
            
    for i in range(len(images)):
        r = images[i].shape[1]/width
        images[i] = resize_img(images[i], width)
        for g in groundtruth[i] :
            g[1:] = [(p/r).astype(int) for p in g[1:]]
            
        
def ensure_dir(file_path):
    """Check a filepath, if doesn't exist, create the directories"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
            
def split_train_test(images, groundtruth, folder_store, part = 0.1):
    '''
    Split a given set of images and groundtruth into a training and a testing set depending on the "part" parameter
    
    ------------Params---------------
    :images : set of given image
    :groundtruth : set of corresponding gt
    :folder_store : where to store the resulting split (if doesn't exist, will create the folder and a img_train/test gt_train/test folder)
    :part : proportion of the dataset for the test
    ''' 
    for i, img in enumerate(images) :
        r = random.random()
        if(len(groundtruth[i] != 0)) : 
            if r > part:
                filepath = "%s/gt_train/%s.txt"%(folder_store, f"{i:04}")
                ensure_dir(filepath)
                f = open(filepath, "w")
                for gt in groundtruth[i] :
                    f.write(to_string(gt) + '\n')
                f.close() 
                filepath = "%s/img_train/%s.jpg"%(folder_store, f"{i:04}")
                ensure_dir(filepath)
                result = cv.imwrite(filepath, img)
            else :
                filepath = "%s/gt_test/%s.txt"%(folder_store, f"{i:04}")
                ensure_dir(filepath)
                f = open(filepath, "w")
                for gt in groundtruth[i] :
                    f.write(to_string(gt) + '\n')
                f.close()  
                filepath = "%s/img_test/%s.jpg"%(folder_store, f"{i:04}")
                ensure_dir(filepath)
                result = cv.imwrite(filepath, img)
                
                
def print_exemple(index, images, groundtruth, modified = False, pred = None, display = True) :
    '''
    Given an index, will print the image with the groundtruth and the prediction 

    :param image: given image
    :param int width: desired final width 
    :param int width: desired final height
    :Note : if both are given, height will be ignored to conserve aspect ratio
    :return: copy of the image resized
    ''' 
    copy = images[index].copy()
    
    if pred is not None:
        if pred[index] is not None:
            for (column, row, width, height) in pred[index]:
                cv.rectangle(copy, (column, row), (width, height),
                (0, 255, 0), 2 )
    if modified :
        for (column, row, width, height) in groundtruth[index]:
            cv.rectangle(copy, (column, row), (column+width, row+height),
            (0, 0, 255), 3)
    else :
        if groundtruth[index].size > 0 :
            for (x1, y1, x2, y2) in groundtruth[index][:, 1:]:
                cv.rectangle(copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

    #copy = resize_img(copy, width=680)
    if display :
        cv.imshow("example", copy)
        cv.waitKey(0)  
        cv.destroyAllWindows()  
    return copy


def get_set_pannels(images, groundtruth, folder_store, p = True):
    """For every image in the set, call the get_pannels() methods (see bellow)
    
    ----------Params-------------
    :images : set of given image
    :groundtruth : set of corresponding gt
    :folder_store : where to store the results (if doesn't exist, will create the folder)
    :p : If true, will split the image according to the pannels, else will create random crops
    """
    for i, img in enumerate(images) :
        get_pannels(img, groundtruth[i], folder_store, i, p)
        
        
def get_pannels(img, gt, folder_store, index, p = True):  
    """Extract the pannels (or random crops), get the corresponding faces in every pannel, recompute their groundtruth according to the pannel they are in
    and finally store the results (pannels + new_gt) in the folder_store
    
    ----------Params-------------
    :images : set of given image
    :groundtruth : set of corresponding gt
    :folder_store : where to store the results (if doesn't exist, will create the folder)
    :p : If true, will split the image according to the pannels, else will create random crops
    """
    
    if p:
        pannels = gt[gt[:,0]==8, 1:]
    else :
        max_size = int(min(images[0].shape[0], images[0].shape[1]) * 0.80)
        min_size = int(min(images[0].shape[0], images[0].shape[1]) * 0.30)
        pannels = create_random_crop(15, img.shape[1], img.shape[0], min_size, max_size)
        
    faces = gt[np.logical_or((gt[:,0]==7), (gt[:, 0] == 11)), 1:]
    a = np.array(array_faces_panels(pannels, faces))
    
    for i, (column, row, width, height) in enumerate(pannels):
            index_faces = np.where(a == i)
            new_gt = []
            r = random.random()
            #if len(faces[index_faces] != 0) :
            if(True):
                if r > 0.1:
                    filepath = "%s/gt_train/%s-%s.txt"%(folder_store, f"{index:04}", f"{i:04}")
                    ensure_dir(filepath)
                    f = open(filepath, "w")
                    for face in faces[index_faces] :
                        f.write(to_string(change_coord_line(resize_to_pannel(pannels[i], face))) + '\n')
                    f.close()
                    crop = img[row:height, column:width]  
                    filepath = "%s/img_train/%s-%s.jpg"%(folder_store, f"{index:04}", f"{i:04}")
                    ensure_dir(filepath)
                    result = cv.imwrite(filepath, crop)
                else :
                    filepath = "%s/gt_test/%s-%s.txt"%(folder_store, f"{index:04}", f"{i:04}")
                    ensure_dir(filepath)
                    f = open(filepath, "w")
                    for face in faces[index_faces] :
                        f.write(to_string(change_coord_line(resize_to_pannel(pannels[i], face))) + '\n')
                    f.close()
                    crop = img[row:height, column:width]  
                    filepath = "%s/img_test/%s-%s.jpg"%(folder_store, f"{index:04}", f"{i:04}")
                    ensure_dir(filepath)
                    result = cv.imwrite(filepath, crop)                    

                    
#-------------Helper methods for the get_pannels() method--------------------
def array_faces_panels(pannels, faces):
    result = []
    for face in faces :
        result.append(find_corresponding_panel(pannels, face))
    return result

def find_corresponding_panel(pannels, face):
    for i, p in enumerate(pannels) :
        if(contains(p, face)):
            return i
    return None

def contains(b1, b2):
    return (b1[0] < b2[0]) & (b1[1] < b2[1]) & (b1[2] > b2[2]) & (b1[3] > b2[3])

def resize_to_pannel(pannel, face):
    copy = face.copy()
    copy[0] = face[0] - pannel[0]
    copy[1] = face[1] - pannel[1]
    copy[2] = face[2] - pannel[0]
    copy[3] = face[3] - pannel[1]
    return copy

def change_coord_line(x):
    return [x[0], x[1], abs(x[2] - x[0]), abs(x[3] - x[1])]

def to_string(a) :
    return ' '.join([str(elem) for elem in a]).replace("[", '').replace(",", "").replace(']', ' ')
    
def create_random_crop(nb, img_width, img_height, min_size, max_size) :
    crops = []
    for i in range(nb):
        x1 = random.randint(0, img_width - min_size)
        y1 = random.randint(0, img_height - min_size)
        x2 = random.randint(x1 + min_size, min(x1 + max_size, img_width))
        y2 = random.randint(y1 + min_size, min(y1 + max_size, img_height))
        crops.append((x1, y1, x2, y2))
    return crops
#----------------------------------------------------------------------------


#----------------------data augmentation methods-----------------------------
def add_flip(images, groundtruth):
    l = len(images)
    for i in range(l) :
        new_img, new_gt = flip(images[i], groundtruth[i])
        images.append(new_img)
        groundtruth.append(new_gt)
        
        
def flip(img, gt):
    flippedimage = cv.flip(img, 1)
    copy = gt.copy()
    middle = int(img.shape[1]/2)
    for i, (c, x1, y1, x2, y2) in enumerate(copy):
        new_x1 = middle + (middle - x1)
        new_x2 = middle + (middle - x2)
        copy[i] = (c, new_x2, y1, new_x1, y2)
    return flippedimage, copy

def blur(images):
    for img in images:
        img = cv.GaussianBlur(img, (5, 5), 0)
    
def rewrite_coord(a):
    return [[x[0], x[1], x[2], x[3]] for x in a]

#---------------------Compute metrics methods-----------------------------
def bb_intersection_over_union(boxA, boxB):
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def box_pred(p, b = False) :
    if b :
        return (p[0], p[1], p[0] + p[2], p[1] + p[3])
    else :
        return (p[0], p[1], p[2], p[3])
    
    
def metric_img(index, groundtruth, pred, b_pred = False) :
    
    tp_img = 0
    fp_img = 0
    already_found = []
    if pred[index] is None :
        return 0, 0, len(groundtruth[index])
    for j in range(len(pred[index])):
        max_ = 0
        max_index = -1
        for k in range(len(groundtruth[index])) :
            if k not in already_found :
                iou = bb_intersection_over_union(box_pred(pred[index][j], b_pred), groundtruth[index][k][1:])
                if iou > max_ : 
                    max_ = iou
                    max_index = k
        if max_ > 0.4: 
            tp_img += 1
            already_found.append(max_index)
        else : fp_img += 1
    fn_img = len(groundtruth[index]) - tp_img
                
    return tp_img, fp_img, fn_img


def compute_metrics(groundtruth, pred, b_pred = False):
    
    fp_tot = 0
    tp_tot = 0
    fn_tot = 0

    for i in range(len(pred)):
        tp, fp, fn = metric_img(i, groundtruth, pred, b_pred)
        fn_tot += fn
        tp_tot += tp
        fp_tot += fp
                
    return tp_tot, fp_tot, fn_tot

def prec_rec_f(tp, fp, fn) :
    if (tp + fp + fn) == 0:
        precision = 1
        recall = 1
        f_score = 1
    else :
        precision = (tp/max(1, (tp + fp)))
        recall = (tp/max(1, (tp + fn)))
        f_score = (precision * recall * 2)/max(1, (precision + recall))
    return precision, recall, f_score


def compute_metric_per_comic(groundtruth, pred, b_pred = False):
    folder = "dcm_dataset.git/images/images"
    l = get_nb_files_per_comics(folder)
    results = []
    for i in range(len(l)):
        tp_tot, fp_tot, fn_tot = compute_metrics(groundtruth[sum(l[:i]):sum(l[:i+1])], pred[sum(l[:i]):sum(l[:i+1])], b_pred)
        precision, recall, f_score = prec_rec_f(tp_tot, fp_tot, fn_tot)
        results.append((tp_tot, fp_tot, fn_tot, round(precision, 3), round(recall,3), round(f_score,3)))
    
    return results


#---------------------Prepare MTCNN data-----------------------------

def get_faces(img, gt, folder_store, index, threshold, nb_max_copy, crop_size, pred, nb_neg = 1):    
    faces = [x[1:] for x in gt if (x[0] == 7 or x[0] == 11)]
    
    for i, box in enumerate(faces):     
            crops_pos = reshape_to_crop_size(img, box, crop_size, nb_max_copy)
            
            if len(crops_pos) == 0 :
                continue
            
            r = random.random()
            if r > threshold:
                for j in range(len(crops_pos)) :
                    filepath_pos = "%s/train/pos/%s-%s-%s.jpg"%(folder_store, f"{index:02}", f"{i:02}", f"{j:02}")
                    #filepath_neg = "%s/train/neg/%s-%s-%s.jpg"%(folder_store, f"{index:02}", f"{i:02}", f"{j:02}")
                    ensure_dir(filepath_pos)
                    #ensure_dir(filepath_neg)
                    result = cv.imwrite(filepath_pos, crops_pos[j])
                    #result = cv.imwrite(filepath_neg, crops_neg[j])
            else :
                for j in range(len(crops_pos)) :
                    filepath_pos = "%s/val/pos/%s-%s-%s.jpg"%(folder_store, f"{index:04}", f"{i:04}", f"{j:02}")
                    #filepath_neg = "%s/val/neg/%s-%s-%s.jpg"%(folder_store, f"{index:04}", f"{i:04}", f"{j:02}")
                    ensure_dir(filepath_pos)
                    #ensure_dir(filepath_neg)
                    result = cv.imwrite(filepath_pos, crops_pos[j])
                    #result = cv.imwrite(filepath_neg, crops_neg[j])
                    
    
    if pred is None :
        crops_neg = get_neg_sample(img, gt, len(faces), crop_size)
    else :
        crops_neg = get_false_pos(img, gt, pred, (len(faces) * nb_neg), crop_size)
            
    for i, crop in enumerate(crops_neg) :        
        r = random.random()
        if r > threshold:
            filepath_neg = "%s/train/neg/%s-%s.jpg"%(folder_store, f"{index:02}", f"{i:02}")
            ensure_dir(filepath_neg)
            result = cv.imwrite(filepath_neg, crop)
        else :
            filepath_neg = "%s/val/neg/%s-%s.jpg"%(folder_store, f"{index:04}", f"{i:04}")
            ensure_dir(filepath_neg)
            result = cv.imwrite(filepath_neg, crop)
            

def get_set_faces(images, groundtruth, folder_store, threshold, nb_max_copy, crop_size, pred = None, nb_neg = 1):
    for i, img in enumerate(images) :
        if pred is not None :
            get_faces(img, groundtruth[i], folder_store, i, threshold, nb_max_copy, crop_size, pred[i], nb_neg)
        else : 
            get_faces(img, groundtruth[i], folder_store, i, threshold, nb_max_copy, crop_size, None)
        

def reshape_to_crop_size(img, box, crop_size, nb_max_copy):
    
    (column, row, c_end, r_end) = box
    
    width = c_end - column
    height = r_end - row
    slide = 0 
    crops = []
    if width > height : 
        i = 0
        while(row - slide >= 0 and slide <= width - height and i < nb_max_copy) :
            slide += max(2, int(width/crop_size))
            if row - slide + width > img.shape[0] :
                continue
            if(img[row-slide:row-slide+width, column:column+width].any()):
                crops.append(img[row-slide:row-slide+width, column:column+width])
                i += 1
    else :
        i = 0
        while(column - slide >= 0 and slide <= height - width and i < nb_max_copy) :
            slide += max(2, int(height/crop_size))            
            if column - slide + height > img.shape[1] : continue
            if(img[row:row+height, column-slide:column-slide+height].any()):
                crops.append(img[row:row+height, column-slide:column-slide+height])
                i += 1
    
    if (len(crops) != 0) :
        crops = [cv.resize(crop, (crop_size, crop_size), interpolation=cv.INTER_LINEAR) for crop in crops]

    return crops

def get_neg_sample(img, gt, nb_total, crop_size):
    
    faces = [x[1:] for x in gt if (x[0] == 7 or x[0] == 11)]
    
    max_size = int(min(img.shape[0], img.shape[1]) * 0.20)
    min_size = int(min(img.shape[0], img.shape[1]) * 0.05)
        
    random_crops = create_random_crop(nb_total*5, img.shape[1], img.shape[0], min_size, max_size)
    valid_crop = []
    
    for r_crop in random_crops:
        valid = True
        for face in faces:
            if(bb_intersection_over_union(r_crop, face)) > 0.1 : 
                valid = False
                break
        if(valid):
            valid_crop.append(r_crop)
            
        if(len(valid_crop) >= nb_total):
            break
            
    crops = []
    for (column, row, c_end, r_end) in valid_crop :
        crops.append(img[row:r_end, column:c_end])
        
    if (len(crops) != 0) :
        crops = [cv.resize(crop, (crop_size, crop_size), interpolation=cv.INTER_LINEAR) for crop in crops]
    return crops


def get_false_pos(img, gt, pred, nb_total, crop_size):
    
    faces = [x[1:] for x in gt if (x[0] == 7 or x[0] == 11)]    
    array_false_pos = []
    
    for p in pred:
        false_pos = True
        for face in faces:
            if(bb_intersection_over_union(p, face)) > 0.1 : 
                false_pos = False
                break
        if(false_pos):
            array_false_pos.append(p)
            
        if(len(array_false_pos) >= nb_total):
            break
            
    crops = []
    for (row, column, r_end, c_end) in array_false_pos :
        #print(column, row, c_end, r_end)
        t = reshape_to_crop_size(img, (int(row), int(column), int(r_end), int(c_end)), crop_size, 1)
        #t = img[int(column):int(c_end), int(row):int(r_end)]
        if len(t) != 0 :
            crops.append(t[0])
        
    if (len(crops) != 0) :
        crops = [cv.resize(crop, (crop_size, crop_size), interpolation=cv.INTER_LINEAR) for crop in crops]
    return crops    
