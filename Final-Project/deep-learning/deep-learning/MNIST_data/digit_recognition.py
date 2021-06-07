import logging
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


class Multi_Digit_Conv_Net(object):
    def __init__(self, structure=None, max_nb_digits=5, nb_channels=1, nb_patch=5, nb_classes=10, img_rows=28, img_cols=28):
        self.__dict__.update(locals())

        self.batch_init = None
        self.structure = structure
        self.session = tf.InteractiveSession()
        self._define_placeholders()
        self._initialize_variables()
        self._network()
        self.saver = tf.train.Saver()

    def _add_layer(self, structure, X):
        """
        Adds new layer to the network. At the moment one can add only a convolutional layer
        or a pool layer. Automatic cast of tensor dimensions needs to be implemented.
        :param structure:
        :param X:
        :return:
        """

        def add_conv_layer(params, X):
            W_var = self._weight_variable([params["patch_x"], params["patch_y"], params["channels"], params["depth"]])
            b_var = self._bias_variable([params["depth"]])
            self.W_conv.append(W_var)
            self.b_conv.append(b_var)
            activation = tf.nn.conv2d(X, W_var, strides=[1, 1, 1, 1], padding='SAME') + b_var
            return tf.nn.relu(activation)

        def add_pool_layer(params, X):
            return tf.nn.max_pool(X, ksize=[1, params["side"], params["side"], 1],
                                  strides=[1, params["stride"], params["stride"], 1], padding=params["pad"])

        if structure[0] == "conv":
            return add_conv_layer(structure[1], X)
        elif structure[0] == "pool":
            return add_pool_layer(structure[1], X)

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _define_placeholders(self):
        """
        Defines all the placeholders needed during the computation
        :return:
        """
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_rows, self.img_cols, self.nb_channels])
        self.y = tf.placeholder(tf.int32, shape=[None, self.max_nb_digits])
        self.keep_prob = tf.placeholder(tf.float32)

    def _initialize_variables(self):
        """
        Initializes weigth and bias variables , and correspondent containers
        :return:
        """

        self.W_conv = []
        self.b_conv = []

        self.W_fc1 = self._weight_variable([7 * 7 * self.max_nb_digits * 64, 1024])
        self.b_fc1 = self._bias_variable([1024])

        # FOR THE ONE-HOT CASE
        #self.W_fcfinal = [self._weight_variable([1024, self.nb_classes]) for i in range(self.max_nb_digits)]
        #self.b_fcfinal = [self._bias_variable([self.nb_classes]) for i in range(self.max_nb_digits)]

        self.W_fcfinal = [self._weight_variable([1024, 1]) for i in range(self.max_nb_digits)]
        self.b_fcfinal = [self._bias_variable([1]) for i in range(self.max_nb_digits)]

    def _network(self):
        """
        Builds the network structure.
        :return:
        """
        X = self.x
        for s in self.structure:
            X = self._add_layer(s, X)

        shape_X = X.get_shape()
        h_pool2_flat = tf.reshape(X, [-1, int(shape_X[1]*shape_X[2]*shape_X[3])])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        self.logits = [tf.matmul(h_fc1_drop, self.W_fcfinal[i]) + self.b_fcfinal[i] for i in range(self.max_nb_digits)]



    def model(self):
        """
        Builds the network structure.
        :return:
        """
        X = self.x

        for s in self.structure:
            X = self._add_layer(s, X)

        shape_X = X.get_shape()
        h_pool2_flat = tf.reshape(X, [-1, int(shape_X[1]*shape_X[2]*shape_X[3])])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        return [tf.matmul(h_fc1_drop, self.W_fcfinal[i]) + self.b_fcfinal[i] for i in range(self.max_nb_digits)]

    def loss(self):
        cr = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits[i], self.y[:,i]))
                             for i in range(self.max_nb_digits)]
        return tf.add_n(cr)

    def accuracy(self, predictions, actual):
        #print predictions.shape, np.sum(actual == np.argmax(predictions,2).T)
        #exit()
        #TODO this is the wrong accuracy. We need equality of all the digits not single ones
        return 100.0* np.sum(np.argmax(predictions, 2).T == actual) / (predictions.shape[1] *predictions.shape[0])

    def fit(self, X, y, X_valid=None, y_valid=None, batch_size = 128, nb_epochs = 100,
            p_dropout = 0.5, logging_info = 10):

        """
        Fits the model to the data. We need to study a way to make it compatible with Sklearn.
        :param data:
        :param validation_data:
        :param batch_size:
        :param nb_epochs:
        :param p_dropout:
        :param logging_info:
        :return:
        """

        train_prediction = tf.pack([tf.nn.softmax(self.logits[i]) for i in range(self.max_nb_digits)])
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)

        #loss_vec = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits[i], self.y[:, i]))
        #      for i in range(self.max_nb_digits)]
        #loss = tf.add_n(loss_vec)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.model()[0], self.y[:, 0]))

        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
        self.session.run(tf.initialize_all_variables())

        self.batch_init = 0
        for i in range(nb_epochs):
            X_batch, y_batch = self._next_batch(X, y, batch_size)
            dic = {self.x: X_batch.reshape((-1, self.img_rows, self.img_cols, self.nb_channels)),
                   self.y: y_batch, self.keep_prob: p_dropout}
            _, l, prediction = self.session.run([optimizer, loss, train_prediction], feed_dict=dic)
            if i % logging_info == 0:
                train_accuracy = self.accuracy(prediction, y_batch)
                print "step {}, training accuracy {}, loss {}".format(i, train_accuracy, l)




    def _next_batch(self, X, y, length):
        if (self.batch_init + 1) * length <= len(y):
            init = self.batch_init * length
            fin = (self.batch_init + 1) * length
            self.batch_init += 1
            return X[init: fin], y[init: fin]
        else:
            init = self.batch_init * length
            self.batch_init = 0
            return X[init:], y[init:]

    def score(self, X, y):
        """
        Provides the score for a new set of instances/targets
        :param X:
        :param y:
        :return:
        """
        def update_dic(batch, keep_prob):
            return { self.x: batch[0].reshape((-1, self.img_rows, self.img_cols, self.nb_channels)),
                self.y[0]: batch[1][:, 0, :], self.y[1]: batch[1][:, 1, :],
                self.y[2]: batch[1][:, 2, :], self.y[3]: batch[1][:, 3, :],
                self.y[4]: batch[1][:, 4, :], self.keep_prob: keep_prob}
        score = self.accuracy().eval(feed_dict=update_dic(batch, 1.0))
        return score

    def predict(self, X):
        """
        Provides new predictions given new targets.
        :param X:
        :return:
        """
        #length = np.zeros((len(X[1]), 1))
        #for i in range(len(X[1])):
        #    for j in range(self.max_nb_digits):
        #        length[i, 0] += np.sum(batch[1][i, j, :])

        dic = {self.x: X.reshape((-1, self.img_rows, self.img_cols, self.nb_channels)), self.keep_prob: 1.0}
        output = [self.output[i] for i in range(self.max_nb_digits)]
        output = np.array([o.eval(feed_dict=dic) for o in output])
        print output.shape
        digit_nb = self.output_length.eval(feed_dict=dic)[:, 0]
        print digit_nb


        #for k in range(X.shape[0]):
        #    print output[:,k,:]
        #    print np.argmax(output[:int(digit_nb[k]),k,:], axis=1)
        #    exit()

        y = np.array([np.argmax(output[:int(digit_nb[k]),k,:], axis=1) for k in range(X.shape[0])])
        print y
        print np_utils.to_categorical(y, 10)
        exit()
        return [output[k, :int(digit_nb[k])] for k in range(X.shape[0])]

    def save(self, path="model.ckpt"):
        save_path = self.saver.save(self.session, path)
        print("Model saved in file: %s" % save_path)

    def load(self, path):
        self.saver.restore(self.session, path)
        print("Model restored.")


if __name__ == "__main__":
    import MNIST_data
    from sklearn.cross_validation import train_test_split



    MNIST = MNIST_data.MNIST()
    nb_digits=1
    X, y = MNIST.synthetize_data(nb_examples=1000, output_method="digits", min_length=1,
                                 max_length=nb_digits, scale_std=0.1, center_std=0.5, rotate_std=0.2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.01, random_state=0)


    print X_train.shape, y_train.shape, X_test.shape, y_test.shape

    structure = [("conv", {"patch_x": 5, "patch_y": 5, "depth": 32, "channels" : 1}), ("pool", {"side": 2, "stride": 2, "pad": "SAME"}),
                 ("conv", {"patch_x": 5, "patch_y": 5, "depth": 64, "channels": 32}), ("pool", {"side": 2, "stride": 2, "pad": "SAME"})]

    C = Multi_Digit_Conv_Net(structure=structure, max_nb_digits=nb_digits, img_rows=28, img_cols=28*nb_digits)
    C.fit(X_train, y_train, batch_size=128, nb_epochs=1000)