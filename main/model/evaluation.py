# Project: ObjetcDetOnKeras
# Filename: evaluation
# Author: Ashutosh Singh
# Date: 08.12.18
# Organisation: Open Source
# Email: ashutosh2564@gmail.com


import numpy as np
import main.utils.utils as utils


def evaluate(model, generator, steps, config):
    """evaluates a model on a generator
    
    Arguments:
        model  -- Keras model
        generator  -- The data generator to evaluate
        steps  -- Number of steps to evaluate
        config  -- a squeezedet config file
    
    Returns:
        [type] -- precision, recall, f1, APs for all classes
    """

    
    batches_processed = 0
    
    all_boxes = []
    all_classes = []
    all_scores = []
    all_gts = []

    print("  Metric Evaluation:")
    print("    Predicting on batches...")

    
    for images, y_true in generator:

        y_pred  = model.predict(images)

        boxes , classes, scores = filter_batch(y_pred, config)


        all_boxes.append(boxes)
        all_classes.append(classes)
        all_scores.append(scores)
        all_gts.append(y_true)
        batches_processed+=1

        if batches_processed == steps:
            break


    print("    Computing statistics...")
    precision, recall, f1,  APs = compute_statistics(all_boxes, all_classes, all_scores, all_gts,  config)

    return precision, recall, f1, APs


def filter_batch( y_pred,config):
    """filters boxes from predictions tensor
    
    Arguments:
        y_pred {[type]} -- tensor of predictions
        config {[type]} -- squeezedet config
    
    Returns:
        lists -- list of all boxes, list of the classes, list of the scores
    """




    pred_class_probs, pred_conf, pred_box_delta = utils.slice_predictions_np(y_pred, config)
    det_boxes = utils.boxes_from_deltas_np(pred_box_delta, config)

    probs = pred_class_probs * np.reshape(pred_conf, [config.BATCH_SIZE, config.ANCHORS, 1])
    det_probs = np.max(probs, 2)
    det_class = np.argmax(probs, 2)



    num_detections = 0


    all_filtered_boxes = []
    all_filtered_scores = []
    all_filtered_classes = [ ]

    for j in range(config.BATCH_SIZE):

        filtered_bbox, filtered_score, filtered_class = filter_prediction(det_boxes[j], det_probs[j],
                                                                          det_class[j], config)


        keep_idx = [idx for idx in range(len(filtered_score)) if filtered_score[idx] > float(config.FINAL_THRESHOLD)]

        final_boxes = [filtered_bbox[idx] for idx in keep_idx]

        final_probs = [filtered_score[idx] for idx in keep_idx]

        final_class = [filtered_class[idx] for idx in keep_idx]


        all_filtered_boxes.append(final_boxes)
        all_filtered_classes.append(final_class)
        all_filtered_scores.append(final_probs)


        num_detections += len(filtered_bbox)


    return all_filtered_boxes, all_filtered_classes, all_filtered_scores


def compute_statistics(all_boxes, all_classes, all_scores, all_gts, config):
    """Computes statistics of all predictions
    
    Arguments:
        all_boxes {[type]} -- list of predicted boxes
        all_classes {[type]} -- list of predicted classes
        all_scores {[type]} --list of predicted scores  
        all_gts {[type]} -- list of all y_trues
        config {[type]} -- squeezedet config
    
    Returns:
        [type] --  prec, rec, f1, APs for all classes
    """


    boxes_per_img, boxes_per_gt, all_tps, all_fps, all_fns, is_gt, all_scores = \
    compute_statistics_for_thresholding(all_boxes, all_classes, all_scores, all_gts, config)


    prec = precision(tp=np.sum(all_tps,axis=0), fp=np.sum(all_fps,axis=0))
    rec = recall(tp=np.sum(all_tps, axis=0), fn=np.sum(all_fns,axis=0))


    f1 = 2 * prec * rec / (prec+rec+1e-20)

    APs, precs, iprecs = AP(is_gt, all_scores)


    print("    Objects {} of {} detected with {} predictions made".format(np.sum(all_tps), np.sum(boxes_per_gt), np.sum(boxes_per_img)))
    for i, name in enumerate(config.CLASS_NAMES):
        print("    Class {}".format(name))
        print("      Precision: {}  Recall: {}".format(prec[i], rec[i]))
        print("      AP: {}".format(APs[i,1]))



    return prec, rec, f1, APs


def compute_statistics_for_thresholding(all_boxes, all_classes, all_scores, all_gts, config):
    """Compute tps, fps, fns, and other stuff for computing APs
    
    
    Arguments:
        all_boxes {[type]} -- list of predicted boxes
        all_classes {[type]} -- list of predicted classes
        all_scores {[type]} --list of predicted scores  
        all_gts {[type]} -- list of all y_trues
        config {[type]} -- squeezedet config
    
    Returns:
        [type] -- boxes_per_img , boxes_per_gt, np.stack(all_tps), np.stack(all_fps), np.stack(all_fns), is_gt, all_score_thresholds
    """



    boxes_per_img = []
    boxes_per_gt = []

    all_tps = []
    all_fps = []

    all_fns = []
    all_score_thresholds = [ [] for c in range(config.CLASSES) ]
    is_gt = [ [] for c in range(config.CLASSES) ]


    for i in range(len(all_boxes)):

        batch_gt = all_gts[i]

        batch_classes = all_classes[i]

        batch_scores = all_scores[i]

        box_input = batch_gt[:, :, 1:5]
        labels = batch_gt[:, :, 9:]

        for j in range(len(all_boxes[i])):

            boxes_per_img.append(len(all_boxes[i][j]))

            non_zero_idx = np.sum(box_input[j][:], axis=-1) > 0

            nonzero_gts = np.reshape(box_input[j][non_zero_idx], [-1,4])

            boxes_per_gt.append(len(nonzero_gts))


            labels_per_image = labels[j]


            nonzero_labels = [ tuple[0]  for labels in  labels_per_image[non_zero_idx,:].astype(int) for tuple in enumerate(labels) if tuple[1]==1  ]

            tp_per_image = np.zeros(config.CLASSES)
            fp_per_image = np.zeros(config.CLASSES)
            fn_per_image = np.zeros(config.CLASSES)



            assigned_idx = np.zeros_like(batch_classes[j])

            for k in range(len(nonzero_gts)):

                try:
                    ious = utils.batch_iou(np.stack(all_boxes[i][j]), nonzero_gts[k])

                    current_score = -1
                    current_idx = -1

                    for iou_index, iou in enumerate(ious):
                        if iou > config.IOU_THRESHOLD \
                        and batch_classes[j][iou_index] == nonzero_labels[k] \
                        and not assigned_idx[iou_index]\
                        and batch_scores[j][iou_index] > current_score:

                            current_score  = batch_scores[j][iou_index]
                            current_idx = iou_index

                    if current_score < 0:
                        fn_per_image[nonzero_labels[k]] += 1

                        is_gt[nonzero_labels[k]].append(1)
                        all_score_thresholds[nonzero_labels[k]].append(0)
                    else:
                        tp_per_image[nonzero_labels[k]] += 1
                        assigned_idx[current_idx] = 1
                        is_gt[nonzero_labels[k]].append(1)
                        all_score_thresholds[nonzero_labels[k]].append(current_score)
                   


                except:

                    fn_per_image[nonzero_labels[k]] = len(nonzero_gts[k])



            for index, ai in enumerate(assigned_idx):

                if ai == 0:

                    fp_per_image[batch_classes[j][index]] +=1
                    is_gt[batch_classes[j][index]].append(0)
                    all_score_thresholds[batch_classes[j][index]].append(batch_scores[j][index])



            all_tps.append(tp_per_image)
            all_fns.append(fn_per_image)
            all_fps.append(fp_per_image)


    return boxes_per_img , boxes_per_gt, np.stack(all_tps), np.stack(all_fps), np.stack(all_fns), is_gt, all_score_thresholds

def AP( predictions, scores):
    """
    Computes the  average precision per class, the average precision and the interpolated average precision at 11 points
    :param predictions: list of lists of every class with tp, fp and fn. fps are zeros, the others one, indicating this is a ground truth
    :param scores: confidences scores with the same lengths
    :return: mAPs a classes x 2 matrix, first entry is without interpolation.
    The average precision and the interpolated average precision at 11 points
    """

    recalls = np.arange(0,1.1,0.1)

    prec = np.zeros_like(recalls)

    iprec =  np.zeros_like(recalls)

    ap = np.zeros( ( len(predictions), 2))

    for i in range(len(predictions)):


        if len(predictions[i]) == 0:
            ap[i,0] = 0
            ap[i,1] = 0

        else:

            zipped = zip(predictions[i], scores[i])


            spreds_and_scores = sorted(zipped, key=lambda x: x[1], reverse=True)

            spreds, sscores = zip(*spreds_and_scores)

            npos = [ t[0] for t in enumerate(spreds) if t[1] > 0 ]


            N = len(npos)

            nprec = np.arange(1,N+1) / (np.array(npos)+1)

            ap[i, 0] = np.mean(nprec)

            inprec =  np.zeros_like(nprec)

            mx = nprec[-1]

            inprec[-1] = mx

            for j in range(len(npos)-2, -1, -1):

                if nprec[j] > mx:
                    mx = nprec[j]
                inprec[j] = mx

            ap[i,1] = np.mean(inprec)


            idx =  (np.concatenate( (np.zeros((1)), np.maximum(np.zeros(10), np.around((N-1)/(10) * np.arange(1,11))-1)))).astype(int)


            iprec += inprec[idx]
            prec += nprec[idx]


    return ap, prec / len(predictions), iprec / len(predictions)



def filter_prediction(boxes, probs, cls_idx, config):
    """Filter bounding box predictions with probability threshold and
    non-maximum supression.
    
    Args:
      boxes: array of [cx, cy, w, h].
      probs: array of probabilities
      cls_idx: array of class indices
    Returns:
      final_boxes: array of filtered bounding boxes.
      final_probs: array of filtered probabilities
      final_cls_idx: array of filtered class indices
    """

    if config.TOP_N_DETECTION < len(probs) and config.TOP_N_DETECTION > 0:
      order = probs.argsort()[:-config.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
      
    else:

      filtered_idx = np.nonzero(probs>config.PROB_THRESH)[0]
      probs = probs[filtered_idx]
      boxes = boxes[filtered_idx]
      cls_idx = cls_idx[filtered_idx]
    
    final_boxes = []
    final_probs = []
    final_cls_idx = []

    for c in range(config.CLASSES):
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]

      keep = utils.nms(boxes[idx_per_class], probs[idx_per_class], config.NMS_THRESH)
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)

    return final_boxes, final_probs, final_cls_idx


def precision(tp,fp):
    """Computes precision for an array of true positives and false positives
    
    Arguments:
        tp {[type]} -- True positives
        fp {[type]} -- False positives
    
    Returns:
        [type] -- Precision
    """

    return tp / (tp+fp+1e-10)


def recall(tp,fn):
    """Computes recall  for an array of true positives and false negatives
    
    Arguments:
        tp {[type]} -- True positives
        fn {function} -- False negatives
    
    Returns:
        [type] -- Recalll
    """
    return tp / (tp+fn+1e-10)



