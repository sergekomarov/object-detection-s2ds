import numpy as np
import math, os, shutil, glob
import bisect
import collections
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps
import xml.etree.ElementTree as ET
import copy

from object_detection.utils import label_map_util
from object_detection.metrics import coco_tools
from object_detection.utils import np_box_ops as npops

from object_detection.utils.np_box_list_ops import multi_class_non_max_suppression
from object_detection.utils.np_box_list import BoxList

from src.preproc import split_image_to_squares


def load_image_into_numpy_array(image_path, target_size,
                                reshape='scale_min_and_crop'):
    """
    Load an image for testing/evaluation into a numpy array. If target_size
    is given, then resize. Also corrects for the EXIF rotation.

    Args:
        image_path (str): path to the image used for testing/evaluation
        target_size (int): see below
        reshape (str):
            'stretch2square': images are stretched to square shape (target_size,target_size);
            'scale_min_and_crop': images are scaled as to keep their aspect ratios, such
                that the smallest dimension becomes target_size;
            'scale_max': images are scaled such that the largest dimension becomes
                target_size
    Return:
        image (np.array): a numpy array of the loaded image
    """
  
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    # remove alpha channel (4th dimension) for cropped test images
    if np.array(image).shape[2]==4:
      image = Image.fromarray(np.array(image)[:, :, 0:3])

    if reshape=='stretch2square':
        return np.array(image.resize((target_size,target_size)))
    elif reshape=='scale_min_and_crop':
        f = target_size / min(image.size)
        return np.array(image.resize((round(f*image.size[0]),
                                      round(f*image.size[1]))))
    elif reshape=='scale_max':
        f = target_size / max(image.size)
        return np.array(image.resize((round(f*image.size[0]),
                                      round(f*image.size[1]))))
    else:
        return np.array(image)

def make_detections(image_np, detect_fn):
    """
    Make detections on an image numpy array using a detection model.
    """
    return detect_fn(tf.convert_to_tensor(image_np)[tf.newaxis, ...])

def postprocess_detections(detections_tf, min_score_thresh=0.5):
    """
    Preprocess raw TensorFlow detections by converting them to numpy arrays
    and removing low-confidence predictions.

    Args
        detections_tf: TF tensor output of a prediction model
        min_score_thresh (float): confidence level below which predictions are discarded
    """
    num_detections = len([x for x in detections_tf['detection_scores'].numpy()[0]
                          if x>min_score_thresh])
    detections_tf.pop('num_detections')
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections_tf.items()}
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    detections['num_detections'] = num_detections
    return detections

def detections_from_image(image_path, detect_fn, target_size,
                          label_map_path,
                          min_score_thresh=0.,
                          reshape='scale_min_and_crop',
                          only_top_left=False):
    """
    When the input image is rectangular (but with the ratio of sides <2) and
    reshape is set to 'scale_min' (scale input images as to keep aspect ratio
    so that the smallest dimension becomes target_size), the function
    splits the image in two intersecting square crops, makes predictions for both,
    then joins them using non-max suppression.

    Otherwise (if the image is a square or reshape='scale_max'), the image is
    fed directly to the detector model after resizing.

    Args
        image_path (str): path to the image used for testing/evaluation
        detect_fn (tf.saved_model): tensorflow loaded model
        target_size (int): size of the final resized image (see argument of load_image_into_numpy_array())
        label_map_path (str): path to the label map
        min_score_thresh (float): confidence level below which predictions are discarded
        reshape (str): how to reshape the input image (see load_image_into_numpy_array())
        only_top_left (bool): make detections for the upper (for vertical images)
            or left (for horizontal) crops only (takes effect only when reshape='scale_min')
    Returns
        Tuple of the image numpy array and a dictionary with detections
    """

    image_np = load_image_into_numpy_array(image_path, target_size, reshape=reshape)

    if max(image_np[:,:,0].shape)/min(image_np[:,:,0].shape) > 2:
        raise ValueError("Cannot process images with tha ratio of largest to" +
            " smallest dimension more than 2")

    # If the resized image is a square or if rescaled by reshape='scale_max',
    # feed the image to the model directly.
    if (image_np.shape[0]==image_np.shape[1]) or (reshape=='scale_max'):
        return image_np, postprocess_detections(make_detections(image_np, detect_fn),
                                                min_score_thresh)

    # Split an image into 2 square crops, save the position of the upper-left
    # corner of the second crop in the original image.
    image0_np, image1_np, pos1, _,_ = split_image_to_squares(image_np)

    # make detections
    detections0 = postprocess_detections(make_detections(image0_np, detect_fn),
                                         min_score_thresh)
    h_px, w_px = image_np.shape[:2]
    h0_px, w0_px = image0_np.shape[:2]

    for i in range(len(detections0['detection_boxes'])):
        ymin,xmin,ymax,xmax = detections0['detection_boxes'][i]
        detections0['detection_boxes'][i][0] = (ymin * h0_px) / h_px
        detections0['detection_boxes'][i][1] = (xmin * w0_px) / w_px
        detections0['detection_boxes'][i][2] = (ymax * h0_px) / h_px
        detections0['detection_boxes'][i][3] = (xmax * w0_px) / w_px

    if not only_top_left:

        detections1 = postprocess_detections(make_detections(image1_np, detect_fn),
                                             min_score_thresh)
        h1_px, w1_px = image1_np.shape[:2]

        # shift predictions for the second crop by its position in the original image
        for i in range(len(detections1['detection_boxes'])):
            ymin,xmin,ymax,xmax = detections1['detection_boxes'][i]
            detections1['detection_boxes'][i][0] = (ymin * h1_px + pos1[1]) / h_px
            detections1['detection_boxes'][i][1] = (xmin * w1_px + pos1[0]) / w_px
            detections1['detection_boxes'][i][2] = (ymax * h1_px + pos1[1]) / h_px
            detections1['detection_boxes'][i][3] = (xmax * w1_px + pos1[0]) / w_px

        boxlist = BoxList(np.concatenate((detections0['detection_boxes'],
                                          detections1['detection_boxes'])))
        classes_all = np.concatenate((detections0['detection_classes'],
                                      detections1['detection_classes']))
        scores_all = np.concatenate((detections0['detection_scores'],
                                    detections1['detection_scores']))

        label_map_dict = label_map_util.get_label_map_dict(str(label_map_path))
        num_classes = len(label_map_dict)
        num_boxes = boxlist.num_boxes()
        scores_all_per_class = np.zeros((num_boxes,num_classes))
        for i in range(num_boxes):
            scores_all_per_class[i, classes_all[i]-1] = scores_all[i]

        boxlist.add_field('classes', classes_all)
        boxlist.add_field('scores', scores_all_per_class)

        # Apply non-max suppression to remove duplicate predictions
        # Always use 0.5 IOU threshold for non-max suppression
        boxlist_sup = multi_class_non_max_suppression(
            boxlist, min_score_thresh,
            0.3, max_output_size=len(detections1['detection_boxes']))

        detections_all = {}
        detections_all['detection_boxes'] = boxlist_sup.data['boxes']
        detections_all['detection_scores'] = boxlist_sup.data['scores']
        detections_all['detection_classes'] = boxlist_sup.data['classes'].astype(np.int64)+1
        detections_all['num_detections'] = len(detections_all['detection_boxes'])
        return image_np, detections_all

    else:
        return image_np, detections0

def detections_from_image_dir(image_dir, detect_fn, target_size, label_map_path,
                              min_score_thresh=0.,
                              reshape='scale_min_and_crop', only_top_left=False):

    detections_from_dir = {}
    image_names = [x for x in os.listdir(image_dir) if (('.jpg' in x) or ('.JPG' in x))]

    for image_name in tqdm(image_names):
        image_path = os.path.join(image_dir, image_name)
        detections_from_dir[image_name] = detections_from_image(
                    image_path, detect_fn, target_size,
                    label_map_path,
                    min_score_thresh=min_score_thresh,
                    reshape=reshape,
                    only_top_left=only_top_left)
    return detections_from_dir

def gt_from_xml(path_to_xml_file, label_map_path,
                read_normalized_coordinates = False):
    """
    Read ground truth annotations from an XML file.

    Args
        path_to_xml_file (str): path to the XML
        label_map_path (str): path to the label map
    Return:
        objects (dict): dictionary of ground truth detections with keys 'detection_boxes',
                'detection_classes'
    """
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    tree = ET.parse(path_to_xml_file)
    root = tree.getroot()
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)

    objects = {'detection_boxes': [], 'detection_classes': []}

    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        ymin = bndbox.find('ymin').text
        xmin = bndbox.find('xmin').text
        ymax = bndbox.find('ymax').text
        xmax = bndbox.find('xmax').text

        if read_normalized_coordinates:
            value = [ float(ymin), float(xmin), float(ymax), float(xmax) ]
        else:
            value = [ int(ymin)/height, int(xmin)/width,
                      int(ymax)/height, int(xmax)/width ]
        name_code = label_map_dict[member.find('name').text]
        objects['detection_boxes'].append(value)
        objects['detection_classes'].append(name_code)
    for key in objects.keys():
        objects[key] = np.array(objects[key])
    objects['detection_boxes'] = objects['detection_boxes'].astype('float32')
    objects['detection_classes'] = objects['detection_classes'].astype('int64')
    return objects

def count_fn(true_fn_flags, ground_truth_classes, class_code):
    """
    Count how many true FN are left in true_fn_flags for class given by class_code

    Args
        true_fn_flags: list of flags that are left 1 if ground truth has not been detected at all
        ground_truth_classes: list of classes corresponding to the ground truths in true_fn_flags
        class_code: code of class that is of interest
    Returns:
        number of 1s left true_fn_flags that correspond to class given by class_code
    """
    count = 0
    for i in range(len(true_fn_flags)):
        if true_fn_flags[i] == 1 and ground_truth_classes[i] == class_code:
            count += 1

    return count

def delete_low_score_detections(detections, min_score_thresh):
    num_detections = len([x for x in detections['detection_scores']
                          if x>min_score_thresh])
    detections.pop('num_detections')
    detections = {key: value[:num_detections]
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    return detections

def get_tp_fp_for_image(image_path,
                        image_predictions,
                        label_map_path,
                        iou_thresh=0.5):
    """
    Classify predictions as True/False Positive for an image given its path.

    Args
        image_path (str): path to the evaluated image
        image_predictions (dict): detections dictionary for the image
        label_map_path (str): path to the label map file
        iou_thresh (float): IOU threshold to identify a matching prediction
    Returns:
        predictions_status (defaultdict): dictionary that assigns a detection
            score, predicted class and TP/FP label to each valid prediction;
        gt_num_AB (int): number of ground truth boxes for Allen Bradley;
        gt_num_S (int):  number of ground truth boxes for Siemens;
    """

    # path to the XML file corresponding to the image at 'image_path'
    gt_xml_path = os.path.splitext(image_path)[0]+'.xml'
    ground_truths = gt_from_xml(gt_xml_path, label_map_path)

    predictions = image_predictions

    image_name = os.path.split(image_path)[1]
    ious = npops.iou(ground_truths['detection_boxes'],predictions['detection_boxes'])
    ground_truth_flags = np.ones(np.shape(ious)[0])
    true_fn_flags = np.ones(np.shape(ious)[0]).astype('int')
    predictions_status = collections.defaultdict(list)
    for prediction in range(np.shape(ious)[1]):
        tmp_flag = 0
        gt_flag = 0
        for ground_truth in range(np.shape(ious)[0]):
            if ious[ground_truth,prediction] > iou_thresh:
                if ground_truths['detection_classes'][ground_truth] == predictions['detection_classes'][prediction] \
                and ground_truth_flags[ground_truth] == 1:
                    tmp_flag += 1
                    ground_truth_flags[ground_truth] -= 1
                    true_fn_flags[ground_truth] -= 1
                else:
                    gt_flag = ground_truths['detection_classes'][ground_truth]
                    true_fn_flags[ground_truth] -= 1
        if tmp_flag == 1:
            predictions_status[image_name+'_pred_'+str(prediction)] = \
                        [predictions['detection_scores'][prediction], \
                         predictions['detection_classes'][prediction],'TP']
        elif tmp_flag == 0:
            predictions_status[image_name+'_pred_'+str(prediction)] = \
                        [predictions['detection_scores'][prediction], \
                         predictions['detection_classes'][prediction],'FP'+str(gt_flag)]
        else:
            print('WARNING! There are most likely overlapping boxes and non-max suppression is not working at IOU thresh: ',
                  iou_thresh)
    if any(x<0 for x in ground_truth_flags):
        print('WARNING! Same object has served as ground truth for multiple TP!')
    category_num = len(label_map_util.get_label_map_dict(label_map_path))
    return (predictions_status,
            [list(ground_truths['detection_classes']).count(x+1) for x in range(category_num)],
            [count_fn(true_fn_flags,ground_truths['detection_classes'], x+1) for x in range(category_num)])

def recall_v_precision(predictions_status, ground_truth_num):
    """
    Classify predictions as True/False Positive for an image given its path.

    Args
        predictions_status: list of predictions classified with TP/FP label
        for all images of interest which have been sorted by prediction scores
        ground_truth_num: number of total groundtruths
    Returns:
        rec_v_prec: list of 2-elements lists consisting of recall v predictions
        points along the path of sorted predictions
    """
    rec_v_prec = []
    tp_num = 0
    fp_num = 0
    for ps in predictions_status:
        if ps[2] == 'TP':
            tp_num += 1
        elif ps[2][:2:] == 'FP':
            fp_num += 1
        rec_v_prec.append([tp_num/ground_truth_num, tp_num/(tp_num+fp_num)])
    return np.array(rec_v_prec)

def AP(rec_v_prec):
    """
    Calculates 101-interpolated Average Precision given recall v precision curve

    Args
        rec_v_prec: list of 2-elements lists consisting of recall v predictions
        points along the path of sorted predictions
    Returns:
        AP: Average Precision
    """
    AP = 0
    for i in np.arange(101)/100:
        ind = bisect.bisect_left(rec_v_prec[:,0],i)
        if ind < len(rec_v_prec[:,0]):
            AP += np.max(rec_v_prec[ind::,1])
    return AP/101

def get_metrics(image_dir, detections_from_dir, label_map_path, 
                min_score_thresh=0.5, iou_thresh=0.5):
    """
    Calculate evaluation metrics for images in a given folder.

    Args
        detections_from_dir (dict): a dictionary whose keys are image names
            in a given directory, and the values are tuples of an image array and
            its detections dictionary
        label_map_path: path to the label map file
        min_score_thresh: minimum confidence score to discard predictions
        iou_thresh (float): IOU threshold to identify a matching prediction
    Return:
        metrics (dict): metrics

    """
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    label_map_dict_rev = {value: key for key, value in label_map_dict.items()}
    ground_truths_nums = np.zeros(len(label_map_dict_rev.keys()))
    true_fn_nums = np.zeros(len(label_map_dict_rev.keys()))

    predictions = collections.defaultdict(list)
    recall_v_precisions = collections.defaultdict(list)
    counters = {}
    images = detections_from_dir.keys()

    detections_from_dir_filt = copy.deepcopy(detections_from_dir)

    for image in tqdm(images):

        detections_from_image = detections_from_dir_filt[image][1]
        detections_from_image = delete_low_score_detections(
            detections_from_image, 
            min_score_thresh
        )

        tmp_metric, tmp_tot_nums, tmp_fn_nums = get_tp_fp_for_image(
                    os.path.join(image_dir,image),
                    detections_from_image,
                    label_map_path,
                    iou_thresh
                    )

        for value in tmp_metric.values():
            predictions[label_map_dict_rev[value[1]]].append(value)
        ground_truths_nums += tmp_tot_nums
        true_fn_nums += tmp_fn_nums

    tmp_count = 0
    for key in label_map_dict.keys():
        predictions[key] = sorted(predictions[key], reverse = True, key = lambda x: x[0])
        recall_v_precisions[key] = recall_v_precision(predictions[key], ground_truths_nums[tmp_count])
        tmp_count += 1
        counters[key] = collections.Counter(np.array(predictions[key]).flatten())

    metrics = collections.defaultdict(int)
    tmp_count = 0
    for key in label_map_dict.keys():
        metrics['AP_'+key] = AP(recall_v_precisions[key])
        metrics[key+'_TP_num'] = counters[key]['TP']
        metrics[key+'_FN_num'] = int(true_fn_nums[tmp_count])
        tmp_count += 1
        metrics[key+'_FP0_num'] = counters[key]['FP0']
        for key2 in label_map_dict_rev.keys():
            if not(key == label_map_dict_rev[key2]):
                metrics[key+'_FP'+str(key2)+'_num'] = counters[key]['FP'+str(key2)]
    return metrics

def check_if_detection_from_xml(xml_path, subcategory=None):
    """
    Check if a .xml file contains any objects of given category - needed to check
    if image contains any groundtruths.
    """
    objects = ET.parse(xml_path).getroot().findall('object')
    if subcategory is not None:
        objects = [x for x in objects if x.find('name').text==subcategory]
    return bool(len(objects))

def create_img_ids_dict(gt_dir, subcategory=None):
    """
    Create dictionary assigning boolean of whether groundtruths are present to each image id. This
    is required for the COCO metrics API.

    Args:
        gt_dir: path to folder containing groundtruth .xml files
        subcategory: if given only count images that contain groundtruths of that category
        as 'True'
    Returns:
        image_ids: dictionary assigning bools to image ids in image_dir
    """
    image_ids = {}
    for image in os.listdir(gt_dir):
        if ('.jpg' in image) or ('.JPG' in image):
            image_ids[os.path.splitext(image)[0]] = check_if_detection_from_xml(
                    os.path.join(gt_dir, os.path.splitext(image)[0]+'.xml'), subcategory)
    return image_ids

def create_groundtruth_list(gt_dir, label_map_path, target_size,
                            reshape='scale_min_and_crop', subcategory=None):
    """
    Create groundtruth_list required by COCO metrics API from groundtruth .xml.

    Args:
        gt_dir: path to folder containing groundtruth .xml files
        label_map_path: path to .pbtxt containing label map information
        target_size: target size of preprocessed test images
        reshape: how test images are reshaped during preprocessing (see load_image_into_numpy_array())
        subcategory: if given will only add groundtruths matching given category
    Returns:
        groundtruth_list: list of dictionaries for each groundtruth containing
        relevant data to calculate COCO metrics
    """
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    groundtruth_list = []
    id_count = 1
    for xml in os.listdir(gt_dir):
        if '.xml' in xml:
            root = ET.parse(os.path.join(gt_dir, xml)).getroot()
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            mins = min(width, height)
            maxs = max(width, height)
            if reshape=='scale_min_and_crop':
                wf = target_size / mins
                hf = wf
            elif reshape=='scale_max':
                wf = target_size / maxs
                hf = wf
            elif reshape=='stretch2square':
                wf = target_size / width
                hf = target_size / height

            for member in root.findall('object'):
                bndbox = member.find('bndbox')
                value = [
                    int(bndbox.find('xmin').text) * wf,
                    int(bndbox.find('ymin').text) * hf,
                    (int(bndbox.find('xmax').text)-int(bndbox.find('xmin').text)) * wf,
                    (int(bndbox.find('ymax').text)-int(bndbox.find('ymin').text)) * hf]
                class_name=member.find('name').text
                if subcategory is not None:
                    if subcategory == class_name:
                        name_code = label_map_dict[class_name]
                        groundtruth_list.append({'id': id_count,
                                                 'image_id': os.path.splitext(xml)[0],
                                                 'category_id': name_code,
                                                 'bbox': value,
                                                 'area': value[2] * value[3],
                                                 'iscrowd': False
                                                 })
                        id_count += 1
                else:
                    name_code = label_map_dict[member.find('name').text]
                    groundtruth_list.append({'id': id_count,
                                             'image_id': os.path.splitext(xml)[0],
                                             'category_id': name_code,
                                             'bbox': value,
                                             'area': value[2]*value[3],
                                             'iscrowd': False
                                             })
                    id_count += 1
    return groundtruth_list

def convert_bbox_coords(bbox_coords, height, width):
    # incoming format: [ymin, xmin, ymax, xmax] normalized
    # needed format: [xmin, ymin, xwidth, yheight] not normalized
    return [bbox_coords[1]*width,
            bbox_coords[0]*height,
            (bbox_coords[3]-bbox_coords[1])*width,
            (bbox_coords[2]-bbox_coords[0])*height
           ]

def create_detection_boxes_list(detections_from_dir):
    """
    Create detection_boxes_list required by COCO metrics API using detections
    from a directory with images.

    Args:
        detections_from_dir (dict): a dictionary whose keys are image names
            in a given directory, and the values are tuples of an image array and
            its detections dictionary
        NOTE: min_score_thresh should be set to 0 when generating detections_from_dir
            in order to keep all the predictions, which is necessary for calculating
            the COCO metrics
    Returns:
        detection_boxes_list: list of dictionaries for each prediction containing
        relevant data to calculate COCO metrics
    """

    detection_boxes_list = []
    images = detections_from_dir.keys()

    for image in images:
        image_np, detections = detections_from_dir[image]
        height, width, channels = np.shape(image_np)
        for i in range(detections['num_detections']):
            detection_boxes_list.append({
                'image_id': os.path.splitext(image)[0],
                'category_id': detections['detection_classes'][i],
                'bbox': convert_bbox_coords(detections['detection_boxes'][i], height, width),
                'score': detections['detection_scores'][i]
            })

    return detection_boxes_list


def create_groundtruth_dict(gt_dir, label_map_path, target_size,
                            reshape='scale_min_and_crop', subcategory=None):
    """
    Create groundtruth dictionary from .xml groundtruths and .pbtxt label map
    """

    image_ids = create_img_ids_dict(gt_dir, subcategory)
    category_index = label_map_util.create_category_index_from_labelmap(
        label_map_path, use_display_name=True)
    categories = [x for x in category_index.values()]
    if subcategory is not None:
        categories = [x for x in categories if x['name']==subcategory]
    groundtruth_list = create_groundtruth_list(gt_dir, label_map_path,
                                               target_size, reshape,
                                               subcategory)

    groundtruth_dict = {
        'annotations': groundtruth_list,
        'images': [{'id': image_id} for image_id in image_ids],
        'categories': categories
    }
    return groundtruth_dict

def get_coco_metrics_from_gt_and_det(groundtruth_dict, detection_boxes_list, category=''):
    """
    Get COCO metrics given dictionary of groundtruth dictionary and the list of
    detections.
    """
    coco_wrapped_groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
    coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(detection_boxes_list)
    box_evaluator = coco_tools.COCOEvalWrapper(coco_wrapped_groundtruth, coco_wrapped_detections, agnostic_mode=False)
    box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(
        include_metrics_per_category=False,
        all_metrics_per_category=False,
        super_categories=None
    )
    box_metrics.update(box_per_category_ap)
    box_metrics = {'DetectionBoxes_'+ category + key: value
                   for key, value in iter(box_metrics.items())}
    return box_metrics

def filter_det_list(det_list, label_map_dict, subcategory):
    """
    Filter out those detections that do not belong to subcategory.
    """
    return [x for x in det_list if x['category_id']==label_map_dict[subcategory]]

def get_coco_metrics(gt_dir, detections_from_dir,
                     label_map_path,
                     target_size,
                     reshape,
                     include_metrics_per_category = False):
    """
    Get COCO metrics given a folder with images and ground truth annotations,
    a detection model, and a label map.

    Args:
        gt_dir: path to folder containing images with groundtruth .xml annotations
            for each image
        detections_from_dir: a dictionary whose keys are image names
            in 'gt_dir', and the values are tuples of an image array and its
            detections dictionary
        label_map_path: path to .pbtxt containing label map information
        target_size: target size of preprocessed test images
        reshape: how test images are reshaped during preprocessing (see load_image_into_numpy_array())
        include_metrics_per_category: if True, COCO metrics will be calculated for each category

    Returns:
        Dictionary containing all the COCO metrics - which are also printed during calculation.
    """

    det_list = create_detection_boxes_list(detections_from_dir)

    if include_metrics_per_category:
        coco_metrics = {}
        label_map_dict = label_map_util.get_label_map_dict(label_map_path)
        for subcategory in label_map_dict.keys():
            gt_dict = create_groundtruth_dict(gt_dir, label_map_path, target_size, reshape, subcategory)
            det_list_tmp = filter_det_list(det_list, label_map_dict, subcategory)
            coco_metrics.update(get_coco_metrics_from_gt_and_det(gt_dict, det_list, category=subcategory+'_'))
        gt_dict = create_groundtruth_dict(gt_dir, label_map_path, target_size, reshape)
        coco_metrics.update(get_coco_metrics_from_gt_and_det(gt_dict, det_list))
        return coco_metrics
    else:
        gt_dict = create_groundtruth_dict(gt_dir, label_map_path, target_size, reshape)
        return get_coco_metrics_from_gt_and_det(gt_dict, det_list)
