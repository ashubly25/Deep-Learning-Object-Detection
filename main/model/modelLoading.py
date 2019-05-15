# Project: ObjetcDetOnKeras
# Filename: modelLoading
# Author: Ashutosh Singh
# Date: 21.12.18
# Organisation: Open Source
# Email: ashutosh2564@gmail.com

import h5py
import numpy as np


def load_only_possible_weights(model, weights_file, verbose=False):
    """
    Sets the weights of a model manually by layer name, even if the length of each dimensions does not match.
    :param model: a keras model
    :param weights_file: a keras ckpt file
    """

    f = h5py.File(weights_file, 'r')

    kernels_and_biases_list = []

    def append_name(name):
        if "kernel" in name or "bias" in name:
            kernels_and_biases_list.append(name)
    f.visit(append_name)

    for l in model.layers:

        w_and_b = l.get_weights()

        if len(w_and_b) == 2:

            for kb in kernels_and_biases_list:
                if l.name + "/kernel" in kb:

                    if verbose:
                        print("Loaded weights for {}".format(l.name))

                    model_shape = np.array(w_and_b[0].shape)
                    file_shape = np.array(f[kb][()].shape)

                    assert len(model_shape) == len(file_shape)


]                    min_dims= np.minimum(model_shape, file_shape)

]                    min_idx = tuple(slice(0, x) for x in min_dims)

                    w_and_b[0][min_idx] = f[kb][()][min_idx]


                if l.name + "/bias" in kb:
                    if verbose:
                        print("Loaded biases for {}".format(l.name))

                    model_shape = np.array(w_and_b[1].shape)
                    file_shape = np.array(f[kb][()].shape)

                    assert len(model_shape) == len(file_shape)

                    min_dims = np.minimum(model_shape, file_shape)

                    min_idx = tuple(slice(0, x) for x in min_dims)

                    w_and_b[1][min_idx] = f[kb][()][min_idx]



            l.set_weights(w_and_b)
