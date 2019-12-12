#!/usr/bin/env python
# Script to train and test a neural network with TF's Keras API

import os
import sys
import argparse
import datetime
import numpy as np
import tensorflow as tf
import saddle_function_utils as sfu

import matplotlib as mpl
from matplotlib import cm
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

from sklearn.utils import shuffle
from sklearn.datasets import make_moons, make_circles, make_classification

from mpl_toolkits.mplot3d import Axes3D

import time

def compute_normalization_parameters(data):
    """
    Compute normalization parameters (mean, st. dev.)
    :param data: matrix with data organized by rows [N x num_variables]
    :return: mean and standard deviation per variable as row matrices of dimension [1 x num_variables]
    """
    # TO-DO. Remove the two lines below and complete this function by computing the real mean and std. dev for the data

    mean = np.mean(data, axis=0)
    stdev = np.std(data, axis=0)

    mean = np.reshape(mean, (1,2))
    stdev = np.reshape(stdev, (1,2))

    return mean, stdev


def normalize_data_per_row(data, mean, stdev):
    """
    Normalize a give matrix of data (samples must be organized per row)
    :param data: input data
    :param mean: mean for normalization
    :param stdev: standard deviation for normalization
    :return: whitened data, (data - mean) / stdev
    """

    # sanity checks!
    assert len(data.shape) == 2, "Expected the input data to be a 2D matrix"
    assert data.shape[1] == mean.shape[1], "Data - Mean size mismatch ({} vs {})".format(data.shape[1], mean.shape[1])
    assert data.shape[1] == stdev.shape[1], "Data - StDev size mismatch ({} vs {})".format(data.shape[1], stdev.shape[1])

    # TODO. Complete. Replace the line below with code to whitten the data.
    normalized_data = (data - mean) / stdev

    return normalized_data


def build_nonlinear_model(num_inputs):
    """
    Build NN model with Keras
    :param num_inputs: number of input features for the model
    :return: Keras model
    """
    # TO-DO: Complete. Remove the None line below, define your model, and return it.

    # input = tf.keras.layers.Input(shape=(num_inputs,), name="inputs")
    # hidden1 = tf.keras.layers.Dense(64, activation='relu')(input)
    # hidden2 = tf.keras.layers.Dense(32, activation='relu')(hidden1)
    # hidden4 = tf.keras.layers.Dense(32, activation='relu')(hidden2)
    # hidden5 = tf.keras.layers.Dense(32, activation='relu')(hidden4)
    # hidden3 = tf.keras.layers.Dense(16, activation='relu')(hidden5) #hidden2
    # output = tf.keras.layers.Dense(1)(hidden3)
    # model = tf.keras.models.Model(inputs=input, outputs=output, name="monkey_model")

    input = tf.keras.layers.Input(shape=(num_inputs,), name="inputs")
    hidden1 = tf.keras.layers.Dense(1280, activation='relu')(input)
    hidden2 = tf.keras.layers.Dense(320, activation='relu')(hidden1)
    #hidden4 = tf.keras.layers.Dense(64, activation='relu')(hidden2)
    hidden5 = tf.keras.layers.Dense(32, activation='relu')(hidden2)
    hidden3 = tf.keras.layers.Dense(16, activation='relu')(hidden5) #hidden2
    #output = tf.keras.layers.Dense(1)(hidden3)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden3)
    model = tf.keras.models.Model(inputs=input, outputs=output, name="monkey_model")

    return model

def build_linear_model(num_inputs):
    """
    Build NN model with Keras
    :param num_inputs: number of input features for the model
    :return: Keras model
    """
    # TO-DO: Complete. Remove the None line below, define your model, and return it.

    input = tf.keras.layers.Input(shape=(num_inputs,), name="inputs")
    hidden1 = tf.keras.layers.Dense(64, use_bias=True)(input)
    output = tf.keras.layers.Dense(1, use_bias=True)(hidden1)
    model = tf.keras.models.Model(inputs=input, outputs=output, name="monkey_model")

    return model


def train_model(model, train_input, train_target, val_input, val_target, input_mean, input_stdev,
                epochs=20, learning_rate=0.01, batch_size=16):
    """
    Train the model on the given data
    :param model: Keras model
    :param train_input: train inputs
    :param train_target: train targets
    :param val_input: validation inputs
    :param val_target: validation targets
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :param epochs: epochs for gradient descent
    :param learning_rate: learning rate for gradient descent
    :param batch_size: batch size for training with gradient descent
    """
    # TO-DO. Complete. Remove the pass line below, and add the necessary training code.
    # normalize

    norm_train_input = normalize_data_per_row(train_input, input_mean, input_stdev)
    norm_val_input = normalize_data_per_row(val_input, input_mean, input_stdev)

    # compile the model: define optimizer, loss, and metrics
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
    #              loss='mse',
    #              metrics=['mae'])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

    # tensorboard callback
    logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)

    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir,'best_monkey_weights.h5'),
                                                            monitor='val_loss',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            mode='auto',
                                                            period=1)

    # do trianing for the specified number of epochs and with the given batch size
    history = model.fit(norm_train_input, train_target, epochs=epochs, batch_size=batch_size,
             validation_data=(norm_val_input, val_target),
             callbacks=[tbCallBack, checkpointCallBack])

    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def test_model(model, test_input, test_target, input_mean, input_stdev, batch_size=60):
    """
    Test a model on a given data
    :param model: trained model to perform testing on
    :param test_input: test inputs
    :param test_target: test targets
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :return: predicted targets for the given inputs
    """
    # TODO. Complete. Remove the return line below and add the necessary code to make predictions with the model.
    norm_test_input = normalize_data_per_row(test_input, input_mean, input_stdev)
    return model.predict(norm_test_input, batch_size=batch_size)



def compute_average_L2_error(test_target, predicted_targets):
    """
    Compute the average L2 error for the predictions
    :param test_target: matrix with ground truth targets [N x 1]
    :param predicted_targets: matrix with predicted targets [N x 1]
    :return: average L2 error
    """
    # TO-DO. Complete. Replace the line below with code that actually computes the average L2 error over the targets.
    temp = test_target-predicted_targets
    average_l2_err = (1.0/test_target.shape[0])*np.linalg.norm(test_target-predicted_targets)

    return average_l2_err


def test_model_output(model, dataX, dataY, mean, stdev, batch_size=60):
    delta = 0.01
    x_min, x_max = dataX[:, 0].min() - .5, dataX[:, 0].max() + .5
    y_min, y_max = dataX[:, 1].min() - .5, dataX[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20),
                         np.linspace(y_min, y_max, 20))
    xxFlat = xx.ravel()
    yyFlat = yy.ravel()
    gridX = np.vstack((xxFlat, yyFlat)).T
    
    norm_input = normalize_data_per_row(gridX, mean, stdev)
    output = model.predict(norm_input, batch_size=batch_size)

    #y0 = np.full(gridX.shape[0], 0.5).reshape((-1, 1))
    #yn, = self.sess.run([self.yn_], feed_dict={self.x_: gridX, self.y0_: y0})
    yn = np.clip(output, 0., 1.)
    zz = 1.-yn.reshape(xx.shape)

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    plt.axis([x_min, x_max, y_min, y_max])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0,top=1,left=0,right=1)
    ax.set_autoscale_on(False)
    ax.grid(False)
    v = np.linspace(0.0, 1.0, 10, endpoint=True)
    plt.contourf(xx, yy, zz, v, alpha=0.5, cmap=cm.bwr)
    # plt.colorbar()
    yFlat = dataY.ravel()
    plt.scatter(dataX[yFlat == 0, 0], dataX[yFlat == 0, 1], color='red')
    plt.scatter(dataX[yFlat == 1, 0], dataX[yFlat == 1, 1], color='blue')
    
    plt.savefig('output.png')
    #for ext in ['png', 'pdf']:
    #    plt.savefig('{}.{}'.format(, ext))
    plt.close()

def main(num_examples, epochs, lr, visualize_training_data, build_fn=build_linear_model, batch_size=16):
    """
    Main function
    :param num_training: Number of examples to generate for the problem (including training, testing, and val)
    :param epochs: number of epochs to train for
    :param lr: learning rate
    :param visualize_training_data: visualize the training data?
    """

    np.random.seed(0) # make the generated values deterministic. do not change!
    
    # generate data
    monkey_function = lambda x: np.power(x[0], 3) - 3*x[0]*np.power(x[1],2)
    input, target = sfu.generate_data_for_2D_function(monkey_function, N=num_examples)

    # split data into training (70%) and testing (30%)
    all_train_input, all_train_target, test_input, test_target = sfu.split_data(input, target, 0.6)

    # visualize all training/testing (uncomment if you want to visualize the whole dataset)
    # plot_train_and_test(all_train_input, all_train_target, test_input, test_target, "train", "test", title="Train/Test Data")

    # split training data into actual training and validation
    train_input, train_target, val_input, val_target = sfu.split_data(all_train_input, all_train_target, 0.8)

    # visualize training/validation (uncomment if you want to visualize the training/validation data)
    if visualize_training_data:
        sfu.plot_train_and_test(train_input, train_target, val_input, val_target, "train", "validation", title="Train/Val Data")

    # normalize input data and save normalization parameters to file
    mean, stdev = compute_normalization_parameters(train_input)

    # build the model
    model = build_fn(train_input.shape[1])

    # train the model
    print("\n\nTRAINING...")
    train_model(model, train_input, train_target, val_input, val_target, mean, stdev,
                epochs=epochs, learning_rate=lr, batch_size=batch_size)

    # test the model
    print("\n\nTESTING...")
    predicted_targets = test_model(model, test_input, test_target, mean, stdev)

    # Report average L2 error
    l2_err = compute_average_L2_error(test_target, predicted_targets)
    print("L2 Error on Testing Set: {}".format(l2_err))

    # visualize the result (uncomment the line below to plot the predictions)
    #sfu.plot_test_predictions(test_input, test_target, predicted_targets, title="Predictions")

    test_model_output(model, test_input, test_target, mean, stdev)



if __name__ == "__main__":

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", help="path to the model",type=str, default="")
    parser.add_argument("--n", help="total number of examples (including training, testing, and validation)",
                        type=int, default=600)
    parser.add_argument("--batch_size", help="batch size used for training",
                        type=int, default=16)
    parser.add_argument("--epochs", help="number of epochs for training",
                        type=int, default=200)
    parser.add_argument("--lr", help="learning rate for training",
                        type=float, default=0.005)
    parser.add_argument("--visualize_training_data", help="visualize training data",
                        action="store_true")
    parser.add_argument("--build_fn", help="model to train (e.g., 'linear')",
                        type=str, default="nonlinear")
    args = parser.parse_args()

    # define the model function that we will use to assemble the Neural Network
    if args.build_fn == "linear":
        build_fn = build_linear_model # function that builds linear model
    elif args.build_fn == "nonlinear":
        build_fn = build_nonlinear_model # function that builds non-linear model
    else:
        print("Invalid build function name {}".format(args.build_fn))
        sys.exit(1)

    if len(args.load_model) > 0:
        build_fn = lambda x: tf.keras.models.load_model(args.load_model, compile=False)


    # run the main function
    t0 = time.time()
    main(args.n, args.epochs, args.lr, args.visualize_training_data, build_fn=build_fn, batch_size=args.batch_size)
    t1 = time.time()

    print(t1-t0)

    sys.exit(0)