import os, sys
import shutil
import glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import xml.etree.ElementTree as ET
from tqdm import tqdm

def clear_dir(dir2clear):
    dir2clear = str(dir2clear)
    if os.path.exists(dir2clear):
        objs = glob.glob(os.path.join(dir2clear, "*"))
        for obj in objs:
            if os.path.isfile(obj):
                os.remove(obj)

def read_label_from_xml(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    classes = []
    bboxes = []

    for boxes in root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None
        c = boxes.find("name").text
        xmin = int(boxes.find("bndbox/xmin").text)
        ymin = int(boxes.find("bndbox/ymin").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        classes.append(c)
        bboxes.append([xmin, ymin, xmax, ymax])

    height = int(root.find("size/height").text)
    width = int(root.find("size/width").text)

    return (classes, bboxes), (height, width)

def write_label_to_xml(xml_file, label, img_shape):

    classes, bboxes = label
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # update image size
    tree.find("size/width").text = str(img_shape[1])
    tree.find("size/height").text = str(img_shape[0])

    # remove objects (number of bboxes after augmentation can be smaller)
    objs = root.findall('object')
    for obj in objs:
        root.remove(obj)

    # add augmented boxes
    for c, bb in zip(classes, bboxes):
        obj = ET.Element('object')
        name = ET.SubElement(obj, 'name')
        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        name.text = str(c)
        xmin.text = str(bb[0])
        ymin.text = str(bb[1])
        xmax.text = str(bb[2])
        ymax.text = str(bb[3])
        root.append(obj)

    tree.write(xml_file)


def copy_augment_data(
    src_dir, dest_dir,
    augment_mult = 1,
    **augment_kwargs):
    """
    Take images and labels from a source directory, apply transformations
    (resize, pad, augment) and copy them to a destination folder.

    Args
        src_dir (str or Path) : source directory
        dest_dir: destination folder where train/validation split is made
        augment_kwargs : augmentation params
        augment_mult (int) : increase the number of images via augmentation by this factor
    """

    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)

    images = ( glob.glob(str(src_dir/'*.jpg'))
             + glob.glob(str(src_dir/'*.JPG')) )
    labels = glob.glob(str(src_dir/'*.xml'))
    images.sort()
    labels.sort()
    # check if the number of images corresponds to the number of .xml files
    assert(len(images)==len(labels))
    n=0
    for i in tqdm(range(len(images))):
        image, label = images[i], labels[i]
        # make sure the copied images have lowercase extensions
        name_base0 = os.path.splitext(os.path.split(image)[1])[0]
        # apply augmentation and copy
        for j in range(augment_mult):
            if j>0: name_base = name_base0 + f"_a{j}"
            else: name_base = name_base0
            make_augmented_image_and_label(
                    image, dest_dir/(name_base+'.jpg'),
                    label, dest_dir/(name_base+'.xml'),
                    **augment_kwargs)
            n += 1
    print(f"\nProduced {n} images with labels")

def make_augmented_image_and_label(
            img_file, img_file_dst,
            lbl_file, lbl_file_dst,
            **aug_kwargs):
    """
    Generate a new image and label by augmenting an existing one.

    Args
        img_file (str): path to the source image
        img_file_dst (str): path to the new image
        lbl_file (str): path to the source label as an .xml file
        lbl_file_dest (str): path to the new image label
        aug_kwargs: arguments for the augmentation function
    """
    img = cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_RGB2BGR)
    lbl, _ = read_label_from_xml(str(lbl_file))
    img_aug, lbl_aug = augment_image_array_and_label(img, lbl, **aug_kwargs)

    cv2.imwrite(str(img_file_dst), cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB))
    if not os.path.exists(lbl_file_dst) or not os.path.samefile(lbl_file, lbl_file_dst):
        shutil.copy(lbl_file, lbl_file_dst)
    write_label_to_xml(str(lbl_file_dst), lbl_aug, img_aug.shape)

def augment_image_array_and_label(img, lbl,
        target_max_size = None,
        pad2square = True,
        rand_augment = False,
        rand_aug_num = 2,
        rand_aug_mag = 1.,
        clip_bb_outside_image = False):
    """
    Apply random augmentations to an image array (including resize/crop/pad)
    and modify bounding boxes correspondingly.

    Args
        img (np.array): input image array
        lbl (tuple): (classes, bboxes)
        target_max_size (int): the largest dimension of the resized image; if
            not given, do not resize
        pad2square (bool): whether to pad images to square shape
        rand_aug_num (int): number of augmentations to apply to an image
        rand_aug_mag (float): magnitude of the augmentations
        clip_bb_outside_image (bool): clip bounding boxes that end up partially
                                      outside the image
    """

    classes, bboxes = lbl
    # form a list of imgaug Bounding Box objects
    _bboxes_im = [BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=c)
                for c,b in zip(classes, bboxes)]
    bboxes_im = BoundingBoxesOnImage(_bboxes_im, shape=img.shape)

    transforms = []

    h = max(img[:,:,0].shape)
    # resize
    if target_max_size is not None:
        factor = target_max_size / h
        transforms.append(iaa.Resize(factor))
    else:
        target_max_size = h

    if pad2square:
        transforms.append(iaa.PadToFixedSize(target_max_size, target_max_size,
                            pad_mode='constant', position='center'))

    if rand_augment:
        mag = rand_aug_mag
        aug_range = (1-0.20*mag, 1+0.20*mag)
        transforms.append(iaa.SomeOf(rand_aug_num,
            [
                iaa.AdditiveGaussianNoise(scale=mag*37, per_channel=True),
                iaa.MotionBlur(k=int(mag*5)),
                iaa.Multiply(aug_range),
                iaa.MultiplyHue(aug_range),
                iaa.MultiplySaturation(aug_range),
                iaa.LinearContrast(aug_range),
                iaa.Sharpen(alpha=0.2*mag, lightness=(0.75, 1.4)),
                iaa.Affine(rotate=(-20*mag, 20*mag)),
                iaa.ShearX(shear=(-15*mag, 15*mag)),
                iaa.ShearY(shear=(-25*mag, 25*mag)),
            ],
            random_order=True
            )
        )

    # apply transformations to the image and its label
    ia_seq = iaa.Sequential(transforms)
    img_aug, bboxes_im_aug = ia_seq(image=img, bounding_boxes=bboxes_im)

    # remove boxes fully outside of the image after transformations
    bboxes_im_aug = bboxes_im_aug.remove_out_of_image()
    # clip those partially outside
    if clip_bb_outside_image:
        bboxes_im_aug = bboxes_im_aug.clip_out_of_image()
    classes = [bb.label for bb in bboxes_im_aug.items]

    bboxes_aug = list(bboxes_im_aug.to_xyxy_array(dtype=np.int32))
    return img_aug, (classes, bboxes_aug)


def split_image_to_squares(img, lbl=None):

    bboxes_im = None
    if lbl is not None:
        classes, bboxes = lbl
        _bboxes_im = [BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=c)
                    for c,b in zip(classes, bboxes)]
        bboxes_im = BoundingBoxesOnImage(_bboxes_im, shape=img.shape)

    min_side = min(img[:,:,0].shape)
    # print(min_side)
    if img.shape[0]>img.shape[1]:
        transform0 = iaa.CropToFixedSize(min_side, min_side,
                                        position='center-bottom')
        transform1 = iaa.CropToFixedSize(min_side, min_side,
                                        position='center-top')
        x1, y1 = 0, img.shape[0] - min_side
        # print('1',x1,y1)
    else:
        transform0 = iaa.CropToFixedSize(min_side, min_side,
                                        position='right-center')
        transform1 = iaa.CropToFixedSize(min_side, min_side,
                                        position='left-center')
        x1, y1 = img.shape[1] - min_side, 0
        # print('2',x1,y1)

    # apply transformations to the image and its label
    img_aug0, bboxes_im_aug0 = transform0(image=img, bounding_boxes=bboxes_im)
    img_aug1, bboxes_im_aug1 = transform1(image=img, bounding_boxes=bboxes_im)

    if lbl is not None:
        bboxes_im_aug0 = bboxes_im_aug0.remove_out_of_image()
        bboxes_im_aug1 = bboxes_im_aug1.remove_out_of_image()
        classes0 = [bb.label for bb in bboxes_im_aug0.items]
        classes1 = [bb.label for bb in bboxes_im_aug1.items]
        bboxes_aug0 = list(bboxes_im_aug0.to_xyxy_array(dtype=np.int32))
        bboxes_aug1 = list(bboxes_im_aug1.to_xyxy_array(dtype=np.int32))
    else:
        bboxes_aug0,bboxes_aug1, classes0,classes1 = None,None, None,None

    return img_aug0, img_aug1, (x1,y1), (classes0, bboxes_aug0), (classes1, bboxes_aug1)

def make_split_images_and_labels(src_dir, dest_dir, only_top_left=False):

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    img_names = [img_name for img_name in os.listdir(src_dir) 
                 if os.path.splitext(img_name)[1].lower()=='.jpg']
    for img_name in tqdm(img_names):
        name = os.path.splitext(img_name)[0]
        img_path = os.path.join(src_dir, img_name)
        lbl_path = os.path.join(src_dir, name+'.xml')
        img_path_dst0 = os.path.join(dest_dir, name+'_0.jpg')
        lbl_path_dst0 = os.path.join(dest_dir, name+'_0.xml')
        img_path_dst1 = os.path.join(dest_dir, name+'_1.jpg')
        lbl_path_dst1 = os.path.join(dest_dir, name+'_1.xml')

        img = cv2.imread(img_path)
        # print('img_shape', img.shape)
        lbl, _ = read_label_from_xml(lbl_path)
        img_aug0, img_aug1, pos1, lbl_aug0, lbl_aug1 = split_image_to_squares(img, lbl)
        # print('img0_shape', img_aug0.shape)
        # print('img1_shape', img_aug1.shape)

        cv2.imwrite(img_path_dst0, img_aug0)
        shutil.copy(lbl_path, lbl_path_dst0)
        write_label_to_xml(lbl_path_dst0, lbl_aug0, img_aug0.shape)
        if not only_top_left:
            cv2.imwrite(img_path_dst1, img_aug1)
            shutil.copy(lbl_path, lbl_path_dst1)
            write_label_to_xml(lbl_path_dst1, lbl_aug1, img_aug1.shape)
