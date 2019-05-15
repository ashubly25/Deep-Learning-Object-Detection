# Project: ObjectDetOnKeras
# Filename: ObjectDet
# Author: Ashutosh Singh
# Date: 28.11.18
# Organisation: Open Source
# Email: ashutosh2564@gmail.com


from keras.models import Model
from keras.layers import Input, MaxPool2D,  Conv2D, Dropout, concatenate, Reshape, Lambda, AveragePooling2D
from keras import backend as K
from keras.initializers import TruncatedNormal
from keras.regularizers import l2
import main.utils.utils as utils
import numpy as np
import tensorflow as tf


class SqueezeDet():
    def __init__(self, config):
        """Init of SqueezeDet Class

        Arguments:
            config {[type]} -- dict containing hyperparameters for network building
        """

        self.config = config
        self.model = self._create_model()

    def _create_model(self):
        """
        #builds the Keras model from config
        #return: squeezeDet in Keras
        """
        input_layer = Input(shape=(self.config.IMAGE_HEIGHT,
                                   self.config.IMAGE_WIDTH, self.config.N_CHANNELS), name="input")

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", activation='relu',
                       use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.001),
                       kernel_regularizer=l2(self.config.WEIGHT_DECAY))(input_layer)

        pool1 = MaxPool2D(pool_size=(3, 3), strides=(
            2, 2), padding='SAME', name="pool1")(conv1)

        fire2 = self._fire_layer(
            name="fire2", input=pool1, s1x1=16, e1x1=64, e3x3=64)

        fire3 = self._fire_layer(
            'fire3', fire2, s1x1=16, e1x1=64, e3x3=64)
        pool3 = MaxPool2D(
            pool_size=(3, 3), strides=(2, 2), padding='SAME', name='pool3')(fire3)

        fire4 = self._fire_layer(
            'fire4', pool3, s1x1=32, e1x1=128, e3x3=128)
        fire5 = self._fire_layer(
            'fire5', fire4, s1x1=32, e1x1=128, e3x3=128)

        pool5 = MaxPool2D(pool_size=(3, 3), strides=(
            2, 2), padding='SAME', name="pool5")(fire5)

        fire6 = self._fire_layer(
            'fire6', pool5, s1x1=48, e1x1=192, e3x3=192)
        fire7 = self._fire_layer(
            'fire7', fire6, s1x1=48, e1x1=192, e3x3=192)
        fire8 = self._fire_layer(
            'fire8', fire7, s1x1=64, e1x1=256, e3x3=256)
        fire9 = self._fire_layer(
            'fire9', fire8, s1x1=64, e1x1=256, e3x3=256)

        fire10 = self._fire_layer(
            'fire10', fire9, s1x1=96, e1x1=384, e3x3=384)
        fire11 = self._fire_layer(
            'fire11', fire10, s1x1=96, e1x1=384, e3x3=384)

        dropout11 = Dropout(rate=self.config.KEEP_PROB, name='drop11')(fire11)

        num_output = self.config.ANCHOR_PER_GRID * \
            (self.config.CLASSES + 1 + 4)

        preds = Conv2D(
            name='conv12', filters=num_output, kernel_size=(3, 3), strides=(1, 1), activation=None, padding="SAME",
            use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.001),
            kernel_regularizer=l2(self.config.WEIGHT_DECAY))(dropout11)

        pred_reshaped = Reshape((self.config.ANCHORS, -1))(preds)

        pred_padded = Lambda(self._pad)(pred_reshaped)

        model = Model(inputs=input_layer, outputs=pred_padded)

        return model

    def _fire_layer(self, name, input, s1x1, e1x1, e3x3, stdd=0.01):
        """
        wrapper for fire layer constructions

        :param name: name for layer
        :param input: previous layer
        :param s1x1: number of filters for squeezing
        :param e1x1: number of filter for expand 1x1
        :param e3x3: number of filter for expand 3x3
        :param stdd: standard deviation used for intialization
        :return: a keras fire layer
        """

        sq1x1 = Conv2D(
            name=name + '/squeeze1x1', filters=s1x1, kernel_size=(1, 1), strides=(1, 1), use_bias=True,
            padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
            kernel_regularizer=l2(self.config.WEIGHT_DECAY))(input)

        ex1x1 = Conv2D(
            name=name + '/expand1x1', filters=e1x1, kernel_size=(1, 1), strides=(1, 1), use_bias=True,
            padding='SAME',  kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
            kernel_regularizer=l2(self.config.WEIGHT_DECAY))(sq1x1)

        ex3x3 = Conv2D(
            name=name + '/expand3x3',  filters=e3x3, kernel_size=(3, 3), strides=(1, 1), use_bias=True,
            padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
            kernel_regularizer=l2(self.config.WEIGHT_DECAY))(sq1x1)

        return concatenate([ex1x1, ex3x3], axis=3)

    def _pad(self, input):
        """
        pads the network output so y_pred and y_true have the same dimensions
        :param input: previous layer
        :return: layer, last dimensions padded for 4
        """

        padding = np.zeros((3, 2))
        padding[2, 1] = 4
        return tf.pad(input, padding, "CONSTANT")

    def loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the total loss
        """

        mc = self.config

        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]
        box_delta_input = y_true[:, :, 5:9]
        labels = y_true[:, :, 9:]

        num_objects = K.sum(input_mask)

        pred_class_probs, pred_conf, pred_box_delta = utils.slice_predictions(
            y_pred, mc)

        det_boxes = utils.boxes_from_deltas(pred_box_delta, mc)

        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])

        ious = utils.tensor_iou(utils.bbox_transform(unstacked_boxes_pred),
                                utils.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc
                                )

        class_loss = K.sum(labels * (-K.log(pred_class_probs + mc.EPSILON))
                           + (1 - labels) *
                           (-K.log(1 - pred_class_probs + mc.EPSILON))
                           * input_mask * mc.LOSS_COEF_CLASS) / num_objects

        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask *
                                                        (pred_box_delta - box_delta_input))) / num_objects)

        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS - num_objects)),
                axis=[1]
            ),
        )

        total_loss = class_loss + conf_loss + bbox_loss

        return total_loss

    def bbox_loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the bbox loss
        """

        mc = self.config

        n_outputs = mc.CLASSES + 1 + 4

        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = K.reshape(
            y_pred, (mc.BATCH_SIZE, mc.N_ANCHORS_HEIGHT, mc.N_ANCHORS_WIDTH, -1))

        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_delta_input = y_true[:, :, 5:9]

        num_objects = K.sum(input_mask)

        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASSES

        num_confidence_scores = mc.ANCHOR_PER_GRID+num_class_probs

        pred_conf = K.sigmoid(
            K.reshape(
                y_pred[:, :, :, num_class_probs:num_confidence_scores],
                [mc.BATCH_SIZE, mc.ANCHORS]
            )
        )

        pred_box_delta = K.reshape(
            y_pred[:, :, :, num_confidence_scores:],
            [mc.BATCH_SIZE, mc.ANCHORS, 4]
        )

        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask *
                                                        (pred_box_delta - box_delta_input))) / num_objects)

        return bbox_loss

    def conf_loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the conf loss
        """

        mc = self.config

        n_outputs = mc.CLASSES + 1 + 4

        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = K.reshape(
            y_pred, (mc.BATCH_SIZE, mc.N_ANCHORS_HEIGHT, mc.N_ANCHORS_WIDTH, -1))

        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]

        num_objects = K.sum(input_mask)

        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASSES

        num_confidence_scores = mc.ANCHOR_PER_GRID+num_class_probs

        pred_conf = K.sigmoid(
            K.reshape(
                y_pred[:, :, :, num_class_probs:num_confidence_scores],
                [mc.BATCH_SIZE, mc.ANCHORS]
            )
        )

        pred_box_delta = K.reshape(
            y_pred[:, :, :, num_confidence_scores:],
            [mc.BATCH_SIZE, mc.ANCHORS, 4]
        )

        det_boxes = utils.boxes_from_deltas(pred_box_delta, mc)

        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])

        ious = utils.tensor_iou(utils.bbox_transform(unstacked_boxes_pred),
                                utils.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc
                                )

        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS - num_objects)),
                axis=[1]
            ),
        )

        return conf_loss

    def class_loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the class loss
        """

        mc = self.config

        n_outputs = mc.CLASSES + 1 + 4

        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = K.reshape(
            y_pred, (mc.BATCH_SIZE, mc.N_ANCHORS_HEIGHT, mc.N_ANCHORS_WIDTH, -1))

        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        labels = y_true[:, :, 9:]

        num_objects = K.sum(input_mask)

        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASSES

        pred_class_probs = K.reshape(
            K.softmax(
                K.reshape(
                    y_pred[:, :, :, :num_class_probs],
                    [-1, mc.CLASSES]
                )
            ),
            [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
        )

        class_loss = K.sum((labels * (-K.log(pred_class_probs + mc.EPSILON))
                            + (1 - labels) * (-K.log(1 - pred_class_probs + mc.EPSILON)))
                           * input_mask * mc.LOSS_COEF_CLASS) / num_objects

        return class_loss

    def loss_without_regularization(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the total loss
        """

        mc = self.config

        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]
        box_delta_input = y_true[:, :, 5:9]
        labels = y_true[:, :, 9:]

        num_objects = K.sum(input_mask)

        pred_class_probs, pred_conf, pred_box_delta = utils.slice_predictions(
            y_pred, mc)

        det_boxes = utils.boxes_from_deltas(pred_box_delta, mc)

        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])

        ious = utils.tensor_iou(utils.bbox_transform(unstacked_boxes_pred),
                                utils.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc)

        class_loss = K.sum(labels * (-K.log(pred_class_probs + mc.EPSILON))
                           + (1 - labels) *
                           (-K.log(1 - pred_class_probs + mc.EPSILON))
                           * input_mask * mc.LOSS_COEF_CLASS) / num_objects

        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask *
                                                        (pred_box_delta - box_delta_input))) / num_objects)

        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS - num_objects)),
                axis=[1]
            ),
        )

        total_loss = class_loss + conf_loss + bbox_loss

        return total_loss
