# Project: ObjectDetOnKeras
# Filename: train
# Author: Ashutosh Singh
# Date: 08.12.18
# Organisation:Open Source
# Email: ashutosh2564@gmail.com


from main.model.squeezeDet import SqueezeDet
from main.model.dataGenerator import generator_from_data_path
import keras.backend as K
from keras import optimizers
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from main.model.modelLoading import load_only_possible_weights
from main.model.multi_gpu_model_checkpoint import ModelCheckpointMultiGPU
import argparse
import os
import gc
from keras.utils import multi_gpu_model
import pickle
from main.config.create_config import load_dict

img_file = "img_train.txt"
gt_file = "gt_train.txt"
log_dir_name = './log'
init_file = "imagenet.h5"
EPOCHS = 100
STEPS = None
OPTIMIZER = "default"
CUDA_VISIBLE_DEVICES = "0"
GPUS = 1
PRINT_TIME = 0
REDUCELRONPLATEAU = True
VERBOSE = False

CONFIG = "squeeze.config"


def train():
    """Def trains a Keras model of SqueezeDet and stores the checkpoint after each epoch
    """

    checkpoint_dir = log_dir_name + "/checkpoints"
    tb_dir = log_dir_name + "/tensorboard"

    if tf.gfile.Exists(checkpoint_dir):
        tf.gfile.DeleteRecursively(checkpoint_dir)

    if tf.gfile.Exists(tb_dir):
        tf.gfile.DeleteRecursively(tb_dir)

    tf.gfile.MakeDirs(tb_dir)
    tf.gfile.MakeDirs(checkpoint_dir)

    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()
    with open(gt_file) as gts:
        gt_names = gts.read().splitlines()
    gts.close()

    cfg = load_dict(CONFIG)

    cfg.img_file = img_file
    cfg.gt_file = gt_file
    cfg.images = img_names
    cfg.gts = gt_names
    cfg.init_file = init_file
    cfg.EPOCHS = EPOCHS
    cfg.OPTIMIZER = OPTIMIZER
    cfg.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES
    cfg.GPUS = GPUS
    cfg.REDUCELRONPLATEAU = REDUCELRONPLATEAU

    if GPUS < 2:

        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

    else:

        gpus = ""
        for i in range(GPUS):
            gpus += str(i)+","
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    cfg.BATCH_SIZE = cfg.BATCH_SIZE * GPUS

    nbatches_train, mod = divmod(len(img_names), cfg.BATCH_SIZE)

    if STEPS is not None:
        nbatches_train = STEPS

    cfg.STEPS = nbatches_train

    print("Number of images: {}".format(len(img_names)))
    print("Number of epochs: {}".format(EPOCHS))
    print("Number of batches: {}".format(nbatches_train))
    print("Batch size: {}".format(cfg.BATCH_SIZE))

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    K.set_session(sess)

    squeeze = SqueezeDet(cfg)

    cb = []

    if OPTIMIZER == "adam":
        opt = optimizers.Adam(lr=0.001 * GPUS,  clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR = 0.001 * GPUS
    if OPTIMIZER == "rmsprop":
        opt = optimizers.RMSprop(lr=0.001 * GPUS,  clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR = 0.001 * GPUS

    if OPTIMIZER == "adagrad":
        opt = optimizers.Adagrad(lr=1.0 * GPUS,  clipnorm=cfg.MAX_GRAD_NORM)
        cfg.LR = 1 * GPUS

    else:

        opt = optimizers.SGD(lr=cfg.LEARNING_RATE * GPUS, decay=0, momentum=cfg.MOMENTUM,
                             nesterov=False, clipnorm=cfg.MAX_GRAD_NORM)

        cfg.LR = cfg.LEARNING_RATE * GPUS

        print("Learning rate: {}".format(cfg.LEARNING_RATE * GPUS))

    with open(log_dir_name + '/config.pkl', 'wb') as f:
        pickle.dump(cfg, f, pickle.HIGHEST_PROTOCOL)

    tbCallBack = TensorBoard(log_dir=tb_dir, histogram_freq=0,
                             write_graph=True, write_images=True)

    cb.append(tbCallBack)

    if REDUCELRONPLATEAU:

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, verbose=1,
                                      patience=5, min_lr=0.0)

        cb.append(reduce_lr)

    if VERBOSE:
        print(squeeze.model.summary())

    if init_file != "none":

        print("Weights initialized by name from {}".format(init_file))

        load_only_possible_weights(squeeze.model, init_file, verbose=VERBOSE)

        """
        for layer in squeeze.model.layers:
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if "fire10" in layer.name or "fire11" in layer.name or "conv12" in layer.name:
                    if hasattr(v_arg, 'initializer'):
                        initializer_method = getattr(v_arg, 'initializer')
                        initializer_method.run(session=sess)
                        #print('reinitializing layer {}.{}'.format(layer.name, v))
        

        """

    train_generator = generator_from_data_path(img_names, gt_names, config=cfg)

    if GPUS > 1:

        ckp_saver = ModelCheckpointMultiGPU(checkpoint_dir + "/model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', verbose=0,
                                            save_best_only=False,
                                            save_weights_only=True, mode='auto', period=1)

        cb.append(ckp_saver)

        print("Using multi gpu support with {} GPUs".format(GPUS))

        parallel_model = multi_gpu_model(squeeze.model, gpus=GPUS)
        parallel_model.compile(optimizer=opt,
                               loss=[squeeze.loss], metrics=[squeeze.loss_without_regularization, squeeze.bbox_loss, squeeze.class_loss, squeeze.conf_loss])

        parallel_model.fit_generator(train_generator, epochs=EPOCHS,
                                     steps_per_epoch=nbatches_train, callbacks=cb)

    else:

        ckp_saver = ModelCheckpoint(checkpoint_dir + "/model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', verbose=0,
                                    save_best_only=False,
                                    save_weights_only=True, mode='auto', period=1)
        cb.append(ckp_saver)

        print("Using single GPU")
        squeeze.model.compile(optimizer=opt,
                              loss=[squeeze.loss], metrics=[squeeze.loss_without_regularization, squeeze.bbox_loss, squeeze.class_loss, squeeze.conf_loss])

        squeeze.model.fit_generator(train_generator, epochs=EPOCHS,
                                    steps_per_epoch=nbatches_train, callbacks=cb)

    gc.collect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train squeezeDet model.')
    parser.add_argument("--steps",  type=int,
                        help="steps per epoch. DEFAULT: #imgs/ batch size")
    parser.add_argument("--epochs", type=int,
                        help="number of epochs. DEFAULT: 100")
    parser.add_argument(
        "--optimizer",  help="Which optimizer to use. DEFAULT: SGD with Momentum and lr decay OPTIONS: SGD, ADAM")
    parser.add_argument(
        "--logdir", help="dir with checkpoints and loggings. DEFAULT: ./log")
    parser.add_argument(
        "--img", help="file of full path names for the training images. DEFAULT: img_train.txt")
    parser.add_argument(
        "--gt", help="file of full path names for the corresponding training gts. DEFAULT: gt_train.txt")
    parser.add_argument("--gpu",  help="which gpu to use. DEFAULT: 0")
    parser.add_argument(
        "--gpus", type=int,  help="number of GPUS to use when using multi gpu support. Overwrites gpu flag. DEFAULT: 1")
    parser.add_argument(
        "--init",  help="keras checkpoint to start training from. If argument is none, training starts from the beginnin. DEFAULT: init_weights.h5")
    parser.add_argument("--resume", type=bool,
                        help="Resumes training and does not delete old dirs. DEFAULT: False")
    parser.add_argument("--reducelr", type=bool,
                        help="Add ReduceLrOnPlateu callback to training. DEFAULT: True")
    parser.add_argument("--verbose", type=bool,
                        help="Prints additional information. DEFAULT: False")
    parser.add_argument(
        "--config",   help="Dictionary of all the hyperparameters. DEFAULT: squeeze.config")

    args = parser.parse_args()

    if args.img is not None:
        img_file = args.img
    if args.gt is not None:
        gt_file = args.gt
    if args.logdir is not None:
        log_dir_name = args.logdir
    if args.gpu is not None:
        CUDA_VISIBLE_DEVICES = args.gpu
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.steps is not None:
        STEPS = args.steps
    if args.optimizer is not None:
        OPTIMIZER = args.optimizer.lower()
    if args.init is not None:
        init_file = args.init
    if args.gpus is not None:
        GPUS = args.gpus
    if args.reducelr is not None:
        REDUCELRONPLATEAU = args.reducelr
    if args.verbose is not None:
        VERBOSE = args.verbose
    if args.config is not None:
        CONFIG = args.config

    train()
