import scipy.io
import numpy as np
from keras.utils import np_utils
import h5py
import cv2
import os


class SVHN_Single_Digit(object):
    def __init__(self):
        """
        Loads the data from file
        """
        self.X_train, self.y_train, self.X_test, self.y_test, self.X_extra, self.y_extra = self._load_data()


    def _load_data(self):
        """
        Loads the data on file
        :return:
        """
        X_train = scipy.io.loadmat('../SVHN_case/train_32x32.mat', variable_names='X').get('X')
        y_train = scipy.io.loadmat('../SVHN_case/train_32x32.mat', variable_names='y').get('y')
        X_test = scipy.io.loadmat('../SVHN_case/test_32x32.mat', variable_names='X').get('X')
        y_test = scipy.io.loadmat('../SVHN_case/test_32x32.mat', variable_names='y').get('y')
        X_extra = scipy.io.loadmat('../SVHN_case/extra_32x32.mat', variable_names='X').get('X')
        y_extra = scipy.io.loadmat('../SVHN_case/extra_32x32.mat', variable_names='y').get('y')

        # Maybe put also in the format data function
        y_train[y_train == 10] = 0
        y_test[y_test == 10] = 0
        y_extra[y_extra == 10] = 0

        return X_train, y_train, X_test, y_test, X_extra, y_extra


    def _format_data(self, X, y, format):
        """
        Formats the data to make it ready for conv net
        :param X:
        :param y:
        :param format:
        :return:
        """
        X = X.transpose((3, 0, 1, 2))
        if format == "one_hot":
            y = np_utils.to_categorical(y, 10).reshape((-1,1,10))
        elif format == "digits":
            y = y[:, 0]
        return X, y


    #Still does not work
    def _generate_index(self, train_lim = 400, extra_lim = 200):
        return



    def datasets(self, seed=0, format = "one_hot", output = "pipe", path_pickle = None, free_mem=True):
        np.random.seed(seed)
        #self._generate_index(5, 5)

        valid_index = []
        valid_index2 = []
        train_index = []
        train_index2 = []

        for i in np.arange(10):
            valid_index.extend(np.where(self.y_train[:, 0] == (i))[0][:400].tolist())
            train_index.extend(np.where(self.y_train[:, 0] == (i))[0][400:].tolist())
            valid_index2.extend(np.where(self.y_extra[:, 0] == (i))[0][:200].tolist())
            train_index2.extend(np.where(self.y_extra[:, 0] == (i))[0][200:].tolist())

        np.random.shuffle(valid_index)
        np.random.shuffle(train_index)
        np.random.shuffle(valid_index2)
        np.random.shuffle(train_index2)

        X_valid = np.concatenate((self.X_extra[:, :, :, valid_index2], self.X_train[:, :, :, valid_index]), axis=3)
        y_valid = np.concatenate((self.y_extra[valid_index2, :], self.y_train[valid_index, :]), axis=0)
        X_train = np.concatenate((self.X_extra[:, :, :, train_index2], self.X_train[:, :, :, train_index]), axis=3)
        y_train = np.concatenate((self.y_extra[train_index2, :], self.y_train[train_index, :]), axis=0)

        X_valid, y_valid = self._format_data(X_valid, y_valid, format)
        X_train, y_train = self._format_data(X_train, y_train, format)
        X_test, y_test = self._format_data(self.X_test, self.y_test, format)

        if free_mem:
            self.X_train = None
            self.y_train = None
            self.X_test = None
            self.y_test = None
            self.X_extra = None
            self.y_extra = None


        X_valid = self._to_greyscale(X_valid)
        X_train = self._to_greyscale(X_train)
        X_test = self._to_greyscale(X_test)

        X_valid = self._contrast_normalization(X_valid).astype(np.float32)
        X_train = self._contrast_normalization(X_train).astype(np.float32)
        X_test = self._contrast_normalization(X_test).astype(np.float32)


        if output == "pipe":
            return X_train, y_train.astype(np.int64), X_valid, y_valid.astype(np.int64), X_test, y_test.astype(np.int64)
        elif output == "pickle":
            from six.moves import cPickle as pickle
            f = open(path_pickle, 'wb')
            dic = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_valid': X_valid,
                    'y_valid': y_valid,
                    'X_test': X_test,
                    'y_test': y_test
                }
            pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
            f.close()


    def _to_greyscale(self, data):
        return np.dot(data.astype(float), [[0.2989], [0.5870], [0.1140]])

    def _contrast_normalization(self, data, min_divisor=1e-4):

        mean = np.mean(data, axis=(1, 2), dtype=float)
        std = np.std(data, axis=(1, 2), dtype=float, ddof=1)
        std[std < min_divisor] = 1.
        norm_data = np.zeros(data.shape, dtype=float)

        for i in np.arange(data.shape[0]):
            norm_data[i, :, :] = (data[i, :, :] - mean[i]) / std[i]

        return norm_data


class SVHN_Full(object):

    def __init__(self, folder):

        self.folder = folder
        self.file = h5py.File(os.path.join(folder, "digitStruct.mat"), 'r')
        self.FileNames = self.file['digitStruct']['name']
        self.Bbox = self.file['digitStruct']['bbox']
        self.boxes = self.get_boxes()

    def get_boxes(self):
        boxes = []
        for i in range(len(self.FileNames)):
            box = {"filename" : ''.join([chr(k[0]) for k in self.file[self.FileNames[i][0]].value])}
            for k in self.file[self.Bbox[i].item()].keys():
                c = self.file[self.Bbox[i].item()][k]
                if len(c) == 1:
                    box[k] = [c.value[0][0]]
                else:
                    box[k] = [self.file[c.value[j].item()].value[0][0] for j in range(len(c))]
            boxes.append(box)
        return boxes

    def generate_data(self):
        X = np.ndarray([len(self.boxes), 32, 32, 1], dtype='float32')
        y = np.ones([len(self.boxes), 6], dtype=int) * 10
        for i, b in enumerate(self.boxes):
            if i%100 ==0:
                print "picture " ,i 
            image = cv2.imread(os.path.join(self.folder, b["filename"]))
            nb_digits = len(b["label"])
            if nb_digits > 5:   #We skip cases with more than 5 digits altogether
                continue
            y[i,0] = nb_digits
            for j in range(nb_digits):
                y[i, j+1] = b["label"][j]
                if y[i, j+1] == 10:
                    y[i, j+1] = 0

            y1 = np.min(b["top"]) - 1
            y2 = np.max(np.array(b["top"])+np.array(b["height"])) + 1
            x1 = np.min(b["left"]) - 1
            x2 = np.max(np.array(b["left"]) + np.array(b["width"])) + 1

            if y1 < 0: y1=0
            if x1 < 0: x1=0

            image = np.dot(image, [[0.2989], [0.5870], [0.1140]])
            try:
                image = cv2.resize(image[y1:y2, x1:x2, 0], (32, 32))
            except:
                print image.shape
                print y1, y2 ,x1, x2
                print b
                exit()
            mean = np.mean(image, dtype='float32')
            std = np.std(image, dtype='float32')

            if std < 1e-4: std = 1.
            image = (image - mean) / std
            X[i, :, :, 0] = image
        return X, y



