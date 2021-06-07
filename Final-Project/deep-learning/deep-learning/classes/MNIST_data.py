from tensorflow.examples.tutorials.mnist import input_data
import logging
import matplotlib.pyplot as plt
import numpy as np
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


class MNIST(object):
    """
    Instances of this class hold the MNIST dataset. Methods are used engineered to build
    datasets from them.
    """
    null_number = np.zeros((28, 28))
    def __init__(self, one_hot=True, path='../MNIST_data'):
        """
        Constructor. It downloads and load the dataset from the tensorflow repository.
        Saves the result in the self.data variable.
        :param one_hot: Output format of the target variable in the dataset.
        :param path: Path were to save the files, or load the files if already downloaded
        """
        self.data = input_data.read_data_sets(path, one_hot=one_hot)

    def learning_sets(self):
        X_train = self.data.train._images.reshape((-1, 28, 28))
        X_test = self.data.test._images.reshape((-1, 28, 28))
        X_valid = self.data.validation._images.reshape((-1, 28, 28))

        y_train = self.data.train._labels.reshape((-1, 1, 10))
        y_test = self.data.test._labels.reshape((-1, 1, 10))
        y_valid = self.data.validation._labels.reshape((-1, 1, 10))

        return  X_train, X_valid, X_test, y_train, y_valid, y_test

    def synthetize_data(self, nb_examples, output_method="one_hot",  max_length=5, min_length = 2,
                        rotate_std=0.01, center_std=0.01, scale_std=0.01, seed=None):
        """
        Building syntethic dataset made of sequences of MNIST digits
        :param nb_examples: Number of instances in the dataset
        :param output_method: Output format of the target variable
        :param max_length: Maximum length to the sequence
        :param min_length: Minimum length of the sequence
        :param rotate_std: Variance of the rotation operation on single digits
        :param center_std: Variance of the translation operation on single digits
        :param scale_std: ...
        :param seed: The see of the numpy random generator
        :return: Returns numpy arrays for feature (X) and target (y) variables.
        """
        if seed is not None:
            np.random.seed(seed)
        X = []
        y = []
        for i in range(nb_examples):
            nb_digits = np.random.randint(min_length, max_length + 1)
            image, label = self._random_image(nb_digits, output_method=output_method, max_length=max_length, rotate_std=rotate_std,
                                              center_std=center_std, scale_std=scale_std)
            X.append(image)
            y.append(label)
        X = np.array(X, dtype=np.float32)
        return X.reshape((-1, X.shape[1], X.shape[2], 1)), np.array(y)

    def _random_image(self, nb_digits, output_method, max_length=5, rotate_std=0.0,
                      center_std=0.0, scale_std=0.0, visualize=False, save=False, save_path=None):
        """
        It generates a random sequence of MNIST images digits. We allow for affine operations on the digit, such as rotations
        and translations.
        :param nb_digits:
        :param output_method:
        :param max_length:
        :param rotate_std:
        :param center_std:
        :param scale_std:
        :param visualize:
        :return: Returns the image and the associated target/label vector in the requested format.
        """

        if nb_digits > max_length:
            nb_digits=max_length


        digit_pos = np.random.randint(0, max_length - nb_digits + 1)
        image_left = [self.null_number for i in range(0,  digit_pos)]
        image_right = [self.null_number for i in range(digit_pos + nb_digits, max_length)]

        digits_id = np.random.randint(len(self.data.train._labels), size=nb_digits)

        image = []
        for i in digits_id:
            digit = self.data.train._images[i].reshape((28, 28))
            theta = np.random.normal(0.0, rotate_std, 1)
            shift = np.random.normal(0.0, center_std, 2)
            scale = np.random.normal(1.0 , scale_std, 1)
            digit = self._translate_digit(digit, shift_X=shift[0], shift_y=shift[1])
            digit = self._rotate_digit(digit, angle=theta, scale=scale)
            image.append(digit)
        full_image = np.concatenate(tuple(image_left + image + image_right), axis=1)

        if output_method=="one_hot":
            label = np.zeros(shape=(max_length, 11), dtype=np.int64)
            for i in range(nb_digits):
                pos = np.argmax(self.data.train._labels[digits_id[i]])
                label[i, pos] = 1
            for i in range(nb_digits, max_length):
                label[i, 10] = 1
        elif output_method=="digits":
            label = 10*np.ones(shape=(max_length + 1,), dtype=np.int64)
            label[0] = nb_digits
            for i in range(nb_digits):
                pos = np.argmax(self.data.train._labels[digits_id[i]])
                label[i + 1] = pos
        else:
            label=None


        if visualize:
            plt.imshow(full_image, interpolation='nearest', cmap='Greys_r')
            plt.show()

        if save:
            plt.imshow(full_image, interpolation='nearest', cmap='Greys_r')
            plt.savefig(save_path)
        return full_image, label

    def _rotate_digit(self, image, angle=0, scale= 1.0):
        """
        Rotates an image
        :param image: The image np.array
        :param angle: The angle of rotation
        :param scale: ...
        :return: The transformed image
        """
        (h, w) = image.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle=angle, scale=scale)
        return cv2.warpAffine(image, M, (w, h))


    def _translate_digit(self, image, shift_X = 0.0, shift_y = 0.0):
        """
        Translates an image
        :param image:
        :param shift_X:
        :param shift_y:
        :return:
        """
        M = np.float32([[1, 0, shift_X], [0, 1, shift_y]])
        return cv2.warpAffine(image, M, image.shape)




if __name__ == '__main__':

    data = MNIST()
    #data._random_image(1 ,output_method="one_hot", max_length=1, rotate_std= 1, center_std=0.0001, scale_std=0.0001, save=False, save_path="MNIST4.png")

    X, y = data.synthetize_data(nb_examples=1000, output_method="digits", min_length=1, max_length=5)

    print X.shape, y.shape

    print y[1]

