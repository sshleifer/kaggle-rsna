from collections import defaultdict
import os
import csv
import datetime
import random
import pydicom
import numpy as np
import pandas as pd
from skimage import io
from skimage import measure
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from metrics import iou_loss, iou_bce_loss, mean_iou
from constants import dicom_dir, bbox_path, det_class_path, test_dicom_dir

# from matplotlib import pyplot as plt
# import matplotlib.patches as patches

class generator(keras.utils.Sequence):
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=32, image_size=256,
                 shuffle=True, augment=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()

    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains pneumonia
        if filename in self.pneumonia_locations:
            # loop through pneumonia
            for location in self.pneumonia_locations[filename]:
                # add 1's at the location of the pneumonia
                x, y, w, h = location
                msk[y:y + h, x:x + w] = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk

    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img

    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)

    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)


def create_downsample(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x = keras.layers.MaxPool2D(2)(x)
    return x

def create_resblock(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return keras.layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=2, depth=4):
    # input
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = keras.layers.UpSampling2D(2**depth)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train(train_filenames, pneumonia_locations, valid_filenames, epochs=25):
    # create train and validation generators
    model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)
    model.compile(optimizer='adam',
                  loss=iou_bce_loss,
                  metrics=['accuracy', mean_iou])

    def cosine_annealing(x, lr=0.001):
        return lr * (np.cos(np.pi * x / epochs) + 1.) / 2

    learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)
    folder = dicom_dir
    train_gen = generator(folder, train_filenames, pneumonia_locations, batch_size=32,
                          image_size=256, shuffle=True, augment=True, predict=False)
    valid_gen = generator(folder, valid_filenames, pneumonia_locations, batch_size=32,
                          image_size=256, shuffle=False, predict=False)

    history = model.fit_generator(train_gen, validation_data=valid_gen, callbacks=[learning_rate],
                                  epochs=epochs, workers=4, use_multiprocessing=True)
    return model, history


def get_pneumonia_locations(bbox_path=bbox_path):
    pneumonia_locations = defaultdict(list)
    # load table
    with open(bbox_path, mode='r') as infile:
        # open reader
        reader = csv.reader(infile)
        # skip header
        next(reader, None)
        # loop through rows
        for rows in reader:
            # retrieve information
            filename = rows[0]
            location = rows[1:5]
            pneumonia = rows[5]
            # if row contains pneumonia add label to dictionary
            # which contains a list of pneumonia locations per filename
            if pneumonia == '1':
                # convert string to float to int
                location = [int(float(i)) for i in location]
                # save pneumonia location in dictionary
                pneumonia_locations[filename].append(location)
    return pneumonia_locations


def make_sub(model, test_filenames, folder=test_dicom_dir, thresh=0.5):

    # create test generator with predict flag set to True
    test_gen = generator(folder, test_filenames, None, batch_size=25, image_size=256, shuffle=False,
                         predict=True)

    # create submission dictionary
    submission_dict = {}
    # loop through testset
    for imgs, filenames in test_gen:
        # predict batch of images
        preds = model.predict(imgs)
        # loop through batch
        for pred, filename in zip(preds, filenames):
            # resize predicted mask
            pred = resize(pred, (1024, 1024), mode='reflect')
            # threshold predicted mask
            comp = pred[:, :, 0] > thresh
            # apply connected components
            comp = measure.label(comp)
            # apply bounding boxes
            predictionString = ''
            for region in measure.regionprops(comp):
                # retrieve x, y, height and width
                y, x, y2, x2 = region.bbox
                height = y2 - y
                width = x2 - x
                # proxy for confidence score
                conf = np.mean(pred[y:y + height, x:x + width])
                # add to predictionString
                predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(
                    width) + ' ' + str(height) + ' '
            # add filename and predictionString to dictionary
            filename = filename.split('.')[0]
            submission_dict[filename] = predictionString
        # stop if we've got them all
        if len(submission_dict) >= len(test_filenames):
            break

    # save dictionary as csv file
    sub = pd.DataFrame.from_dict(submission_dict, orient='index')
    sub.index.names = ['patientId']
    sub.columns = ['PredictionString']
    return sub

def make_sub_path(mode):

    dt = str(datetime.datetime.now())
    if mode == 'utest':
        return 'blah.csv'
    else:
        return 'submission_{}.csv'.format(dt)


def main(epochs=1, mode='utest'):
    # load and shuffle filenames
    pneumonia_locations = get_pneumonia_locations()
    folder = dicom_dir
    filenames = os.listdir(folder)
    random.shuffle(filenames)
    # split into train and validation filenames
    if mode == 'utest':
        n_valid_samples=1
        train_filenames = filenames[1:4]
        valid_filenames = filenames[5:6]
    else:
        n_valid_samples = 2560
        train_filenames = filenames[n_valid_samples:]
        valid_filenames = filenames[:n_valid_samples]
    print('n train samples', len(train_filenames))
    print('n valid samples', len(valid_filenames))
    n_train_samples = len(filenames) - n_valid_samples

    model, history = train(train_filenames, pneumonia_locations, valid_filenames, epochs=epochs)
    print(history)
    # load and shuffle filenames

    test_filenames = os.listdir(test_dicom_dir)
    if mode == 'utest':
        test_filenames = test_filenames[:3]

    sub = make_sub(model, test_filenames)
    sub.to_csv(make_sub_path(mode))

    return model, history




if __name__ == '__main__':
    main(epochs=2, mode='utest')
