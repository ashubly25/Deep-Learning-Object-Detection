# Project: ObjectDetOnKeras
# Filename: visualization
# Author: Ashutosh Singh
# Date: 12.12.18
# Organisation: Opensource
# Email: ashutosh2564@gmail.com
from main.model.evaluation import filter_batch
import cv2
import numpy as np


def visualize(model, generator, config):
    """Creates images with ground truth and from the model predicted boxes.

    Arguments:
        model {[type]} -- SqueezeDet Model
        generator {[type]} -- data generator yielding images and ground truth
        config {[type]} --  dict of various hyperparameters

    Returns:
        [type] -- numpy array of images with ground truth and prediction boxes added
    """

    nbatches, mod = divmod(config.VISUALIZATION_BATCH_SIZE, config.BATCH_SIZE)

    print("  Creating Visualizations...")

    count = 0

    all_boxes = []

    for images, y_true, images_only_resized in generator:

        y_pred = model.predict(images)

        images_with_boxes = visualize_dt_and_gt(
            images_only_resized, y_true, y_pred, config)

        try:
            all_boxes.append(np.stack(images_with_boxes))
        except:
            pass

        count += 1

        if count >= nbatches:
            break
    try:
        return np.stack(all_boxes).reshape((-1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
    except:
        return np.zeros((nbatches*config.BATCH_SIZE, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))


def visualize_dt_and_gt(images, y_true, y_pred, config):
    """Takes a batch of images and creates bounding box visualization on top

    Arguments:
        images {[type]} -- numpy tensor of images
        y_true {[type]} -- tensor of ground truth
        y_pred {[type]} -- tensor of predictions
        config {[type]} -- dict of various hyperparameters

    Returns:
        [type] -- dict of various hyperparameters
    """

    img_with_boxes = []

    all_filtered_boxes, all_filtered_classes, all_filtered_scores = filter_batch(
        y_pred, config)

    box_input = y_true[:, :, 1:5]

    labels = y_true[:, :, 9:]

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, img in enumerate(images):

        non_zero_boxes = box_input[i][box_input[i] > 0].reshape((-1, 4))

        non_zero_labels = []

        for k, coords in enumerate(box_input[i, :]):
            if np.sum(coords) > 0:

                for j, l in enumerate(labels[i, k]):
                    if l == 1:
                        non_zero_labels.append(j)

        for j, det_box in enumerate(all_filtered_boxes[i]):

            det_box = bbox_transform_single_box(det_box)

            cv2.rectangle(img, (det_box[0], det_box[1]),
                          (det_box[2], det_box[3]), (0, 0, 255), 1)
            cv2.putText(img, config.CLASS_NAMES[all_filtered_classes[i][j]] + " " + str(
                all_filtered_scores[i][j]), (det_box[0], det_box[1]), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        for j, gt_box in enumerate(non_zero_boxes):

            gt_box = bbox_transform_single_box(gt_box)

            cv2.rectangle(img, (gt_box[0], gt_box[1]),
                          (gt_box[2], gt_box[3]), (0, 255, 0), 1)
            cv2.putText(img, config.CLASS_NAMES[int(non_zero_labels[j])], (gt_box[0], gt_box[1]), font, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

        img_with_boxes.append(img[:, :, [2, 1, 0]])

    return img_with_boxes


def bbox_transform_single_box(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = int(np.floor(cx-w/2))
    out_box[1] = int(np.floor(cy-h/2))
    out_box[2] = int(np.floor(cx+w/2))
    out_box[3] = int(np.floor(cy+h/2))

    return out_box
