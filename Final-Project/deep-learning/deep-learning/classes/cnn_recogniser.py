import tensorflow as tf
import numpy as np

### OLD IMPLEMENTATIONS
class CNN_Digit_Recogniser(object):
    def __init__(self, structure=None, nb_channels=1, nb_classes=10, img_rows=32, img_cols=32, max_nb_digits=5):
        self.__dict__.update(locals())
        self.graph = tf.Graph()
        self.structure = structure
        self.saver = None
        self.logger = {"training_error" : [], "validation_error" : [], "test_error" : []}


    def _define_placeholders(self):
        """
        Defines all the placeholders needed during the computation
        :return:
        """
        self.x = tf.placeholder(
            tf.float32, shape=[None, self.img_rows, self.img_cols, self.nb_channels])
        self.y = tf.placeholder(tf.int64, shape=[None, self.max_nb_digits + 1])

    def _initialize_variables(self):
        """
        Initializes weigth and bias variables , and correspondent containers
        """

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        self.W_conv = []
        self.b_conv = []
        for s in self.structure:
            if s[0] == "conv":
                self.W_conv.append(weight_variable([s[1]["patch_x"], s[1]["patch_y"],
                                                          s[1]["channels"], s[1]["depth"]]))
                self.b_conv.append(bias_variable([s[1]["depth"]]))
                last_depth = s[1]["depth"]

        self.W_fc1 = weight_variable([(self.img_rows/4) * (self.img_cols/4) * last_depth, 1024])
        self.b_fc1 = bias_variable([1024])

        #self.W_fc2 = weight_variable([512, 1024])
        #self.b_fc2 = bias_variable([1024])

        self.W_fcfinal = [weight_variable([1024, self.nb_classes]) for i in range(self.max_nb_digits)]
        self.b_fcfinal = [bias_variable([self.nb_classes]) for i in range(self.max_nb_digits)]

    def _model(self, data_X, dropout):
        """
        Builds the network structure.
        """
        def add_conv_layer(data, W, b):
            activation = tf.nn.conv2d(data, W, strides=[1, 1, 1, 1], padding='SAME') + b
            return tf.nn.local_response_normalization(tf.nn.relu(activation))


        def add_pool_layer(data, params):
            return tf.nn.max_pool(data, ksize=[1, params["side"], params["side"], 1],
                                  strides=[1, params["stride"], params["stride"], 1], padding=params["pad"])
        features = data_X
        conv = 0
        for s in self.structure:
            if s[0] == "conv":
                features = add_conv_layer(features, self.W_conv[conv], self.b_conv[conv])
                conv += 1
            elif s[0] == "pool":
                features = add_pool_layer(features, s[1])

        shape = features.get_shape()
        sub2_flat = tf.reshape(features, [-1, int(shape[1]*shape[2]*shape[3])])
        h_fc1 = tf.nn.relu(tf.matmul(sub2_flat, self.W_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, dropout)

        return [tf.matmul(h_fc1_drop, self.W_fcfinal[i]) + self.b_fcfinal[i] for i in range(self.max_nb_digits)]

    def _accuracy_digits(self, predictions, actual):
        return 100.0 * np.sum(np.argmax(predictions, 2).T == actual[:,1:self.max_nb_digits+1]) \
               / (predictions.shape[0] * predictions.shape[1])

    def _accuracy_full(self, predictions, actual):
        count = 0
        for i in range(actual.shape[0]):
            vec = np.where(actual[i, 1: self.max_nb_digits+1] == 10)[0]
            if len(vec) != 0:
                lim = vec[0]
            else:
                lim  = self.max_nb_digits
            if np.array_equal(actual[i, 1:lim+2],np.argmax(predictions, 2).T[i, :lim+1]):
                count += 1
        return 100.0*count/actual.shape[0]

    def fit(self, X, y, X_valid=None, y_valid=None, X_test=None, y_test=None, batch_size = 128, nb_epochs = 100,
            p_dropout = 0.5, logging_info = 10, save_path=None, seed=None):

        """
        Fits the network using AdagradOptimizer #TODO setting up the parameters for
        :param X: The feature vector for the training set
        :param y: The target vector for the trainin set
        :param X_valid: (Optional) The feature vector for the validation set
        :param y_valid: (Optional) The target vector for the validation set
        :param batch_size: The size of batch fed for each epoch
        :param nb_epochs: The number of epochs in a minibatch framework
        :param p_dropout: Dropout probability (only applied once)
        :param logging_info: An integer managing logging informations
        :param save_path: The path of the saved model. If None, the model is not saved
        :param seed: The seed of the tensorflow random generator. Used to set up the initial weights of the network
        """

        def next_batch(X, y, length):
            if (self.batch_init + 1) * length <= len(y):
                init = self.batch_init * length
                fin = (self.batch_init + 1) * length
                self.batch_init += 1
                return X[init: fin], y[init: fin]
            else:
                init = self.batch_init * length
                self.batch_init = 0
                return X[init:], y[init:]

        def prepare_dict(batch):
            dic = {self.x: batch[0].reshape(-1, self.img_rows, self.img_cols, 1),
                   self.y : batch[1]}
            return dic

        with self.graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._define_placeholders()
            self._initialize_variables()
            self.saver = tf.train.Saver()

            logits = self._model(self.x, p_dropout)

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[0], self.y[:, 1]))
            for i in range(1, self.max_nb_digits):
                loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[i], self.y[:, i+1]))

            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

            train_prediction = tf.pack([tf.nn.softmax(self._model(self.x, 1.0)[i]) for i in range(self.max_nb_digits)])
            if X_valid is not None:
                valid_prediction = tf.pack([tf.nn.softmax(self._model(X_valid, 1.0)[i]) for i in range(self.max_nb_digits)])
            if X_test is not None:
                test_prediction = tf.pack([tf.nn.softmax(self._model(X_test, 1.0)[i]) for i in range(self.max_nb_digits)])


        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            self.batch_init = 0
            for step in range(nb_epochs):
                batch = next_batch(X, y, batch_size)
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=prepare_dict(batch))
                if step % logging_info == 0:
                    print("Minibatch loss value at step {}: {:.2f}".format(step+1, l))
                    minibatch_accuracy_full = self._accuracy_full(predictions, batch[1])
                    minibatch_accuracy_digits = self._accuracy_digits(predictions, batch[1])
                    print("Minibatch digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                        minibatch_accuracy_digits, minibatch_accuracy_full))
                    self.logger["training_error"].append(np.array([minibatch_accuracy_digits, minibatch_accuracy_full]))

                    if X_valid is not None:
                        # Maybe the following line is the problem with increase of size of the checkpoint file
                        valid_pred = valid_prediction.eval(feed_dict={})
                        valid_accuracy_full = self._accuracy_full(valid_pred, y_valid)
                        valid_accuracy_digits = self._accuracy_digits(valid_pred, y_valid)
                        self.logger["validation_error"].append(np.array([valid_accuracy_digits, valid_accuracy_full]))
                        print("Validation set digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                            valid_accuracy_digits, valid_accuracy_full))

            if X_test is not None:
                test_pred = test_prediction.eval(feed_dict={})
                test_accuracy_full = self._accuracy_full(test_pred, y_test)
                test_accuracy_digits = self._accuracy_digits(test_pred, y_test)
                self.logger["test_error"].append(np.array([test_accuracy_digits, test_accuracy_full]))
                print("Test set digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                    test_accuracy_digits, test_accuracy_full))


            if save_path is not None:
                self.saver.save(session, save_path)
                print "Model saved in {}".format(save_path)

    def score(self, X, y, restore_path=None):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            if restore_path is not None:
                self.saver.restore(session, restore_path)
            predictions = tf.pack(
                [tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0)[i]) for i in
                 range(self.max_nb_digits)])
            predictions = predictions.eval(feed_dict={})
            return self._accuracy(predictions, y), self._accuracy_full(predictions, y)

    def predict(self, X, restore_path=None):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            if restore_path is not None:
                self.saver.restore(session, restore_path)
            predictions = tf.pack([tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0)[i]) for i in range(self.max_nb_digits)])
            predictions = predictions.eval(feed_dict={})
            return np.argmax(predictions, 2).T
class CNN_Digit_Recogniser_Mod(object):
    def __init__(self, structure=None, nb_channels=1, nb_classes=10, img_rows=32, img_cols=32, max_nb_digits=5):
        self.__dict__.update(locals())
        self.graph = tf.Graph()
        self.structure = structure
        self.saver = None
        self.logger = {"training_error" : [], "validation_error" : [], "test_error" : []}

    def _define_placeholders(self):
        """
        Defines all the placeholders needed during the computation
        :return:
        """
        self.x = tf.placeholder(
            tf.float32, shape=[None, self.img_rows, self.img_cols, self.nb_channels])
        self.y = tf.placeholder(tf.int64, shape=[None, self.max_nb_digits + 1])

    def _initialize_variables(self):
        """
        Initializes weigth and bias variables , and correspondent containers
        """

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        self.W_conv = []
        self.b_conv = []
        for s in self.structure:
            if s[0] == "conv":
                self.W_conv.append(weight_variable([s[1]["patch_x"], s[1]["patch_y"],
                                                          s[1]["channels"], s[1]["depth"]]))
                self.b_conv.append(bias_variable([s[1]["depth"]]))
                last_depth = s[1]["depth"]

        self.W_fc1 = [weight_variable([(self.img_rows/4) * (self.img_cols/4) * last_depth, 1024])
                      for i in range(self.max_nb_digits)]
        self.b_fc1 = [bias_variable([1024]) for i in range(self.max_nb_digits)]

        #self.W_fc2 = weight_variable([512, 1024])
        #self.b_fc2 = bias_variable([1024])

        self.W_fcfinal = [weight_variable([1024, self.nb_classes]) for i in range(self.max_nb_digits)]
        self.b_fcfinal = [bias_variable([self.nb_classes]) for i in range(self.max_nb_digits)]

    def _model(self, data_X, dropout):
        """
        Builds the network structure.
        """
        def add_conv_layer(data, W, b):
            activation = tf.nn.conv2d(data, W, strides=[1, 1, 1, 1], padding='SAME') + b
            return tf.nn.local_response_normalization(tf.nn.relu(activation))


        def add_pool_layer(data, params):
            return tf.nn.max_pool(data, ksize=[1, params["side"], params["side"], 1],
                                  strides=[1, params["stride"], params["stride"], 1], padding=params["pad"])
        features = data_X
        conv = 0
        for s in self.structure:
            #print features.get_shape(), s
            if s[0] == "conv":
                features = add_conv_layer(features, self.W_conv[conv], self.b_conv[conv])
                conv += 1
            elif s[0] == "pool":
                features = add_pool_layer(features, s[1])

        shape = features.get_shape()
        sub2_flat = tf.reshape(features, [-1, int(shape[1]*shape[2]*shape[3])])


        h_fc1 = [tf.nn.relu(tf.matmul(sub2_flat, self.W_fc1[i]) + self.b_fc1[i]) for i in range(self.max_nb_digits)]
        h_fc1_drop = [tf.nn.dropout(h_fc1[i], dropout) for i in range(self.max_nb_digits)]
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2)
        #h_fc2_drop = tf.nn.dropout(h_fc2, dropout)

        return [tf.matmul(h_fc1_drop[i], self.W_fcfinal[i]) + self.b_fcfinal[i] for i in range(self.max_nb_digits)]

    def _accuracy_digits(self, predictions, actual):
        return 100.0 * np.sum(np.argmax(predictions, 2).T == actual[:,1:self.max_nb_digits+1]) \
               / (predictions.shape[0] * predictions.shape[1])

    def _accuracy_full(self, predictions, actual):
        count = 0
        for i in range(actual.shape[0]):
            vec = np.where(actual[i, 1: self.max_nb_digits+1] == 10)[0]
            if len(vec) != 0:
                lim = vec[0]
            else:
                lim  = self.max_nb_digits
            if np.array_equal(actual[i, 1:lim+2],np.argmax(predictions, 2).T[i, :lim+1]):
                count += 1
        return 100.0*count/actual.shape[0]

    def fit(self, X, y, X_valid=None, y_valid=None, X_test=None, y_test=None, batch_size = 128, nb_epochs = 100,
            p_dropout = 0.5, logging_info = 10, save_path=None, seed=None):

        """
        Fits the network using AdagradOptimizer #TODO setting up the parameters for
        :param X: The feature vector for the training set
        :param y: The target vector for the trainin set
        :param X_valid: (Optional) The feature vector for the validation set
        :param y_valid: (Optional) The target vector for the validation set
        :param batch_size: The size of batch fed for each epoch
        :param nb_epochs: The number of epochs in a minibatch framework
        :param p_dropout: Dropout probability (only applied once)
        :param logging_info: An integer managing logging informations
        :param save_path: The path of the saved model. If None, the model is not saved
        :param seed: The seed of the tensorflow random generator. Used to set up the initial weights of the network
        """

        def next_batch(X, y, length):
            if (self.batch_init + 1) * length <= len(y):
                init = self.batch_init * length
                fin = (self.batch_init + 1) * length
                self.batch_init += 1
                return X[init: fin], y[init: fin]
            else:
                init = self.batch_init * length
                self.batch_init = 0
                return X[init:], y[init:]

        def prepare_dict(batch):
            dic = {self.x: batch[0].reshape(-1, self.img_rows, self.img_cols, 1),
                   self.y : batch[1]}
            return dic

        with self.graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._define_placeholders()
            self._initialize_variables()
            self.saver = tf.train.Saver()

            logits = self._model(self.x, p_dropout)

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[0], self.y[:, 1]))
            for i in range(1, self.max_nb_digits):
                loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[i], self.y[:, i+1]))

            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

            train_prediction = tf.pack([tf.nn.softmax(self._model(self.x, 1.0)[i]) for i in range(self.max_nb_digits)])
            if X_valid is not None:
                valid_prediction = tf.pack([tf.nn.softmax(self._model(X_valid, 1.0)[i]) for i in range(self.max_nb_digits)])
            if X_test is not None:
                test_prediction = tf.pack([tf.nn.softmax(self._model(X_test, 1.0)[i]) for i in range(self.max_nb_digits)])


        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            self.batch_init = 0
            for step in range(nb_epochs):
                batch = next_batch(X, y, batch_size)
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=prepare_dict(batch))
                if step % logging_info == 0:
                    print("Minibatch loss value at step {}: {:.2f}".format(step+1, l))
                    minibatch_accuracy_full = self._accuracy_full(predictions, batch[1])
                    minibatch_accuracy_digits = self._accuracy_digits(predictions, batch[1])
                    print("Minibatch digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                        minibatch_accuracy_digits, minibatch_accuracy_full))
                    self.logger["training_error"].append(np.array([minibatch_accuracy_digits, minibatch_accuracy_full]))

                    if X_valid is not None:
                        # Maybe the following line is the problem with increase of size of the checkpoint file
                        valid_pred = valid_prediction.eval(feed_dict={})
                        valid_accuracy_full = self._accuracy_full(valid_pred, y_valid)
                        valid_accuracy_digits = self._accuracy_digits(valid_pred, y_valid)
                        self.logger["validation_error"].append(np.array([valid_accuracy_digits, valid_accuracy_full]))
                        print("Validation set digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                            valid_accuracy_digits, valid_accuracy_full))

            if X_test is not None:
                test_pred = test_prediction.eval(feed_dict={})
                test_accuracy_full = self._accuracy_full(test_pred, y_test)
                test_accuracy_digits = self._accuracy_digits(test_pred, y_test)
                self.logger["test_error"].append(np.array([test_accuracy_digits, test_accuracy_full]))
                print("Test set digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                    test_accuracy_digits, test_accuracy_full))
                self.test_predictions = test_pred


            if save_path is not None:
                self.saver.save(session, save_path)
                print "Model saved in {}".format(save_path)

    def score(self, X, y, restore_path=None):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            if restore_path is not None:
                self.saver.restore(session, restore_path)
            predictions = tf.pack(
                [tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0)[i]) for i in
                 range(self.max_nb_digits)])
            predictions = predictions.eval(feed_dict={})
            return self._accuracy(predictions, y), self._accuracy_full(predictions, y)

    def predict(self, X, restore_path=None):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            if restore_path is not None:
                self.saver.restore(session, restore_path)
            predictions = tf.pack([tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0)[i]) for i in range(self.max_nb_digits)])
            predictions = predictions.eval(feed_dict={})
            return np.argmax(predictions, 2).T


class Recogniser_Type1(object):
    def __init__(self, structure=None, nb_channels=1, nb_classes=10, img_rows=32, img_cols=32, max_nb_digits=5, nb_hidden=1024):
        self.__dict__.update(locals())
        self.graph = tf.Graph()
        self.structure = structure
        self.saver = None
        self.logger = {"training_error" : [], "validation_error" : [], "test_error" : []}


    def _define_placeholders(self):
        """
        Defines all the placeholders needed during the computation
        :return:
        """
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_rows, self.img_cols, self.nb_channels])
        self.y = tf.placeholder(tf.int64, shape=[None, self.max_nb_digits + 1])

    def _initialize_variables(self):
        """
        Initializes weigth and bias variables , and correspondent containers
        """

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        self.W_conv = []
        self.b_conv = []
        pool_prod = 1
        for s in self.structure:
            if s[0] == "conv":
                self.W_conv.append(weight_variable([s[1]["patch_x"], s[1]["patch_y"],
                                                          s[1]["channels"], s[1]["depth"]]))
                self.b_conv.append(bias_variable([s[1]["depth"]]))
                last_depth = s[1]["depth"]
            if s[0] == "pool":
                pool_prod *= s[1]["side"]

        self.W_fc1 = weight_variable([(self.img_rows/pool_prod) * (self.img_cols/pool_prod) * last_depth, self.nb_hidden])
        self.b_fc1 = bias_variable([self.nb_hidden])


        self.W_fcfinal = [weight_variable([self.nb_hidden, self.nb_classes]) for i in range(self.max_nb_digits)]
        self.b_fcfinal = [bias_variable([self.nb_classes]) for i in range(self.max_nb_digits)]

    def _model(self, data_X, dropout):
        """
        Builds the network structure.
        """
        def add_conv_layer(data, W, b):
            activation = tf.nn.conv2d(data, W, strides=[1, 1, 1, 1], padding='SAME') + b
            return tf.nn.local_response_normalization(tf.nn.relu(activation))


        def add_pool_layer(data, params):
            return tf.nn.max_pool(data, ksize=[1, params["side"], params["side"], 1],
                                  strides=[1, params["stride"], params["stride"], 1], padding='SAME')
        features = data_X
        conv = 0
        for s in self.structure:
            if s[0] == "conv":
                features = add_conv_layer(features, self.W_conv[conv], self.b_conv[conv])
                conv += 1
            elif s[0] == "pool":
                features = add_pool_layer(features, s[1])

        shape = features.get_shape()
        sub2_flat = tf.reshape(features, [-1, int(shape[1]*shape[2]*shape[3])])
        h_fc1 = tf.nn.relu(tf.matmul(sub2_flat, self.W_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, dropout)

        return [tf.matmul(h_fc1_drop, self.W_fcfinal[i]) + self.b_fcfinal[i] for i in range(self.max_nb_digits)]

    def _accuracy_digits(self, predictions, actual):
        return 100.0 * np.sum(np.argmax(predictions, 2).T == actual[:,1:self.max_nb_digits+1]) \
               / (predictions.shape[0] * predictions.shape[1])

    def _accuracy_full(self, predictions, actual):
        count = 0
        for i in range(actual.shape[0]):
            vec = np.where(actual[i, 1: self.max_nb_digits+1] == 10)[0]
            if len(vec) != 0:
                lim = vec[0]
            else:
                lim  = self.max_nb_digits
            if np.array_equal(actual[i, 1:lim+2],np.argmax(predictions, 2).T[i, :lim+1]):
                count += 1
        return 100.0*count/actual.shape[0]

    def fit(self, X, y, X_valid=None, y_valid=None, X_test=None, y_test=None, batch_size = 128, nb_epochs = 100,
            p_dropout = 0.5, logging_info = 10, optimizer={"type" : "Adagrad"}, save_path=None, seed=None):

        """
        Fits the network using AdagradOptimizer #TODO setting up the parameters for
        :param X: The feature vector for the training set
        :param y: The target vector for the trainin set
        :param X_valid: (Optional) The feature vector for the validation set
        :param y_valid: (Optional) The target vector for the validation set
        :param batch_size: The size of batch fed for each epoch
        :param nb_epochs: The number of epochs in a minibatch framework
        :param p_dropout: Dropout probability (only applied once)
        :param logging_info: An integer managing logging informations
        :param save_path: The path of the saved model. If None, the model is not saved
        :param seed: The seed of the tensorflow random generator. Used to set up the initial weights of the network
        """

        def next_batch(X, y, length):
            if (self.batch_init + 1) * length <= len(y):
                init = self.batch_init * length
                fin = (self.batch_init + 1) * length
                self.batch_init += 1
                return X[init: fin], y[init: fin]
            else:
                init = self.batch_init * length
                self.batch_init = 0
                return X[init:], y[init:]

        def prepare_dict(batch):
            dic = {self.x: batch[0].reshape(-1, self.img_rows, self.img_cols, 1),
                   self.y : batch[1]}
            return dic

        def _optimizer_type(optimizer, loss):
            if optimizer["type"] == "Adagrad":
                learning_rate = tf.train.exponential_decay(0.05, tf.Variable(0), 10000, 0.95)
                return tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=tf.Variable(0))
            elif optimizer["type"] == "Adadelta":
                return tf.train.AdadeltaOptimizer(optimizer["learning_rate"]).minimize(loss)

        with self.graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._define_placeholders()
            self._initialize_variables()
            self.saver = tf.train.Saver()

            logits = self._model(self.x, p_dropout)

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[0], self.y[:, 1]))
            for i in range(1, self.max_nb_digits):
                loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[i], self.y[:, i+1]))

            optimizer = _optimizer_type(optimizer, loss)

            train_prediction = tf.pack([tf.nn.softmax(self._model(self.x, 1.0)[i]) for i in range(self.max_nb_digits)])
            if X_valid is not None:
                valid_prediction = tf.pack([tf.nn.softmax(self._model(X_valid, 1.0)[i]) for i in range(self.max_nb_digits)])
            if X_test is not None:
                test_prediction = tf.pack([tf.nn.softmax(self._model(X_test, 1.0)[i]) for i in range(self.max_nb_digits)])


        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            self.batch_init = 0
            for step in range(nb_epochs):
                batch = next_batch(X, y, batch_size)
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=prepare_dict(batch))
                if step % logging_info == 0:
                    print("Minibatch loss value at step {}: {:.2f}".format(step+1, l))
                    minibatch_accuracy_full = self._accuracy_full(predictions, batch[1])
                    minibatch_accuracy_digits = self._accuracy_digits(predictions, batch[1])
                    print("Minibatch digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                        minibatch_accuracy_digits, minibatch_accuracy_full))
                    self.logger["training_error"].append(np.array([minibatch_accuracy_digits, minibatch_accuracy_full]))

                    if X_valid is not None:
                        # Maybe the following line is the problem with increase of size of the checkpoint file
                        valid_pred = valid_prediction.eval(feed_dict={})
                        valid_accuracy_full = self._accuracy_full(valid_pred, y_valid)
                        valid_accuracy_digits = self._accuracy_digits(valid_pred, y_valid)
                        self.logger["validation_error"].append(np.array([valid_accuracy_digits, valid_accuracy_full]))
                        print("Validation set digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                            valid_accuracy_digits, valid_accuracy_full))

            if X_test is not None:
                test_pred = test_prediction.eval(feed_dict={})
                test_accuracy_full = self._accuracy_full(test_pred, y_test)
                test_accuracy_digits = self._accuracy_digits(test_pred, y_test)
                self.logger["test_error"].append(np.array([test_accuracy_digits, test_accuracy_full]))
                print("Test set digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                    test_accuracy_digits, test_accuracy_full))


            if save_path is not None:
                self.saver.save(session, save_path)
                print "Model saved in {}".format(save_path)

    def score(self, X, y, restore_path=None):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            if restore_path is not None:
                self.saver.restore(session, restore_path)
            predictions = tf.pack(
                [tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0)[i]) for i in
                 range(self.max_nb_digits)])
            predictions = predictions.eval(feed_dict={})
            return self._accuracy(predictions, y), self._accuracy_full(predictions, y)

    def predict(self, X, restore_path=None):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            if restore_path is not None:
                self.saver.restore(session, restore_path)
            predictions = tf.pack([tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0)[i]) for i in range(self.max_nb_digits)])
            predictions = predictions.eval(feed_dict={})
            return np.argmax(predictions, 2).T

class Recogniser_Type2(object):
    def __init__(self, structure=None, nb_channels=1, nb_classes=10, img_rows=32, img_cols=32, max_nb_digits=5, nb_hidden=1024):
        self.__dict__.update(locals())
        self.graph = tf.Graph()
        self.structure = structure
        self.saver = None
        self.logger = {"training_error" : [], "validation_error" : [], "test_error" : []}

    def _define_placeholders(self):
        """
        Defines all the placeholders needed during the computation
        :return:
        """
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_rows, self.img_cols, self.nb_channels])
        self.y = tf.placeholder(tf.int64, shape=[None, self.max_nb_digits + 1])

    def _initialize_variables(self):
        """
        Initializes weigth and bias variables , and correspondent containers
        """

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        self.W_conv = []
        self.b_conv = []

        # Works in the case both pool and convolution have SAME PADDING (we force same for now)
        pool_prod = 1
        for s in self.structure:
            if s[0] == "conv":
                self.W_conv.append(weight_variable([s[1]["patch_x"], s[1]["patch_y"],
                                                          s[1]["channels"], s[1]["depth"]]))
                self.b_conv.append(bias_variable([s[1]["depth"]]))
                last_depth = s[1]["depth"]
            if s[0] == "pool":
                pool_prod *= s[1]["side"]

        self.W_fc1 = [weight_variable([(self.img_rows/pool_prod) * (self.img_cols/pool_prod) * last_depth, self.nb_hidden])
                      for i in range(self.max_nb_digits)]
        self.b_fc1 = [bias_variable([self.nb_hidden]) for i in range(self.max_nb_digits)]

        self.W_fcfinal = [weight_variable([self.nb_hidden, self.nb_classes]) for i in range(self.max_nb_digits)]
        self.b_fcfinal = [bias_variable([self.nb_classes]) for i in range(self.max_nb_digits)]

    def _model(self, data_X, dropout):
        """
        Builds the network structure.
        """
        def add_conv_layer(data, W, b):
            activation = tf.nn.conv2d(data, W, strides=[1, 1, 1, 1], padding='SAME') + b
            return tf.nn.local_response_normalization(tf.nn.relu(activation))


        def add_pool_layer(data, params):
            return tf.nn.max_pool(data, ksize=[1, params["side"], params["side"], 1],
                                  strides=[1, params["stride"], params["stride"], 1], padding='SAME')
        features = data_X
        conv = 0
        for s in self.structure:
            if s[0] == "conv":
                features = add_conv_layer(features, self.W_conv[conv], self.b_conv[conv])
                conv += 1
            elif s[0] == "pool":
                features = add_pool_layer(features, s[1])

        shape = features.get_shape()
        sub2_flat = tf.reshape(features, [-1, int(shape[1]*shape[2]*shape[3])])
        h_fc1 = [tf.nn.relu(tf.matmul(sub2_flat, self.W_fc1[i]) + self.b_fc1[i]) for i in range(self.max_nb_digits)]
        h_fc1_drop = [tf.nn.dropout(h_fc1[i], dropout) for i in range(self.max_nb_digits)]

        return [tf.matmul(h_fc1_drop[i], self.W_fcfinal[i]) + self.b_fcfinal[i] for i in range(self.max_nb_digits)]

    def _accuracy_digits(self, predictions, actual):
        return 100.0 * np.sum(np.argmax(predictions, 2).T == actual[:,1:self.max_nb_digits+1]) \
               / (predictions.shape[0] * predictions.shape[1])

    def _accuracy_full(self, predictions, actual):
        count = 0
        for i in range(actual.shape[0]):
            vec = np.where(actual[i, 1: self.max_nb_digits+1] == 10)[0]
            if len(vec) != 0:
                lim = vec[0]
            else:
                lim  = self.max_nb_digits
            if np.array_equal(actual[i, 1:lim+2],np.argmax(predictions, 2).T[i, :lim+1]):
                count += 1
        return 100.0*count/actual.shape[0]

    def fit(self, X, y, X_valid=None, y_valid=None, X_test=None, y_test=None, batch_size = 128, nb_epochs = 100,
            p_dropout = 0.5, logging_info = 10, optimizer={"type" : "Adagrad"} ,save_path=None, seed=None):

        """
        Fits the network using AdagradOptimizer #TODO setting up the parameters for
        :param X: The feature vector for the training set
        :param y: The target vector for the trainin set
        :param X_valid: (Optional) The feature vector for the validation set
        :param y_valid: (Optional) The target vector for the validation set
        :param batch_size: The size of batch fed for each epoch
        :param nb_epochs: The number of epochs in a minibatch framework
        :param p_dropout: Dropout probability (only applied once)
        :param logging_info: An integer managing logging informations
        :param save_path: The path of the saved model. If None, the model is not saved
        :param seed: The seed of the tensorflow random generator. Used to set up the initial weights of the network
        """

        def next_batch(X, y, length):
            if (self.batch_init + 1) * length <= len(y):
                init = self.batch_init * length
                fin = (self.batch_init + 1) * length
                self.batch_init += 1
                return X[init: fin], y[init: fin]
            else:
                init = self.batch_init * length
                self.batch_init = 0
                return X[init:], y[init:]

        def prepare_dict(batch):
            dic = {self.x: batch[0].reshape(-1, self.img_rows, self.img_cols, 1),
                   self.y : batch[1]}
            return dic

        def _optimizer_type(optimizer, loss):
            if optimizer["type"] == "Adagrad":
                learning_rate = tf.train.exponential_decay(0.05, tf.Variable(0), 10000, 0.95)
                return tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=tf.Variable(0))
            elif optimizer["type"] == "Adadelta":
                learning_rate = tf.train.exponential_decay(0.05, tf.Variable(0), 10000, 0.95)
                return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=tf.Variable(0))


        with self.graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._define_placeholders()
            self._initialize_variables()
            self.saver = tf.train.Saver()

            logits = self._model(self.x, p_dropout)

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[0], self.y[:, 1]))
            for i in range(1, self.max_nb_digits):
                loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[i], self.y[:, i+1]))

            optimizer = _optimizer_type(optimizer, loss)

            train_prediction = tf.pack([tf.nn.softmax(self._model(self.x, 1.0)[i]) for i in range(self.max_nb_digits)])
            if X_valid is not None:
                valid_prediction = tf.pack([tf.nn.softmax(self._model(X_valid, 1.0)[i]) for i in range(self.max_nb_digits)])
            if X_test is not None:
                test_prediction = tf.pack([tf.nn.softmax(self._model(X_test, 1.0)[i]) for i in range(self.max_nb_digits)])


        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            self.batch_init = 0
            for step in range(nb_epochs):
                batch = next_batch(X, y, batch_size)
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=prepare_dict(batch))
                if step % logging_info == 0:
                    print("Minibatch loss value at step {}: {:.2f}".format(step+1, l))
                    minibatch_accuracy_full = self._accuracy_full(predictions, batch[1])
                    minibatch_accuracy_digits = self._accuracy_digits(predictions, batch[1])
                    print("Minibatch digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                        minibatch_accuracy_digits, minibatch_accuracy_full))
                    self.logger["training_error"].append(np.array([minibatch_accuracy_digits, minibatch_accuracy_full]))

                    if X_valid is not None:
                        valid_pred = valid_prediction.eval(feed_dict={})
                        valid_accuracy_full = self._accuracy_full(valid_pred, y_valid)
                        valid_accuracy_digits = self._accuracy_digits(valid_pred, y_valid)
                        self.logger["validation_error"].append(np.array([valid_accuracy_digits, valid_accuracy_full]))
                        print("Validation set digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                            valid_accuracy_digits, valid_accuracy_full))

            if X_test is not None:
                test_pred = test_prediction.eval(feed_dict={})
                test_accuracy_full = self._accuracy_full(test_pred, y_test)
                test_accuracy_digits = self._accuracy_digits(test_pred, y_test)
                self.logger["test_error"].append(np.array([test_accuracy_digits, test_accuracy_full]))
                print("Test set digit accuracy: {:.1f}%, full sequence accuracy: {:.1f}%".format(
                    test_accuracy_digits, test_accuracy_full))
                self.test_predictions = test_pred


            if save_path is not None:
                self.saver.save(session, save_path)
                print "Model saved in {}".format(save_path)

    def score(self, X, y, restore_path=None):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            if restore_path is not None:
                self.saver.restore(session, restore_path)
            predictions = tf.pack(
                [tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0)[i]) for i in
                 range(self.max_nb_digits)])
            predictions = predictions.eval(feed_dict={})
            return self._accuracy(predictions, y), self._accuracy_full(predictions, y)

    def predict(self, X, restore_path=None):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            if restore_path is not None:
                self.saver.restore(session, restore_path)
            predictions = tf.pack([tf.nn.softmax(self._model(X.reshape(-1, self.img_rows, self.img_cols, 1), 1.0)[i]) for i in range(self.max_nb_digits)])
            predictions = predictions.eval(feed_dict={})
            return np.argmax(predictions, 2).T

def MNIST():
    import MNIST_data
    from sklearn.cross_validation import train_test_split

    digits = 2
    data = MNIST_data.MNIST()
    X, y = data.synthetize_data(nb_examples=40000, output_method="digits", min_length=1, max_length=digits)

    X, X_test, y, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=0)

    structure = [("conv", {"patch_x": 5, "patch_y": 5, "depth": 32, "channels": 1}),
                 ("pool", {"side": 2, "stride": 2, "pad": "SAME"}),
                 ("conv", {"patch_x": 5, "patch_y": 5, "depth": 64, "channels": 32}),
                 ("pool", {"side": 2, "stride": 2, "pad": "SAME"})]

    seed = 100
    C = Recogniser_Type2(structure=structure, nb_classes=11, img_rows=28, img_cols=28*digits, max_nb_digits=digits)
    C.fit(X, y, X_test, y_test, batch_size=128, nb_epochs=3500, logging_info=100, seed=seed, p_dropout=0.5)


def MNIST_SD():
    import MNIST_data
    MNIST = MNIST_data.MNIST()
    X, X_valid, X_test, y, y_valid, y_test = MNIST.learning_sets()

    X = X.reshape((-1, X.shape[1], X.shape[2], 1))
    X_valid = X_valid.reshape((-1, X_valid.shape[1], X_valid.shape[1], 1))
    X_test = X_test.reshape((-1, X_test.shape[1], X_test.shape[1], 1))

    y = np.argmax(y[:,0,:],1)
    y_test = np.argmax(y_test[:, 0, :], 1)
    y_valid = np.argmax(y_valid[:, 0, :], 1)

    structure = [("conv", {"patch_x": 5, "patch_y": 5, "depth": 32, "channels": 1}),
                 ("pool", {"side": 2, "stride": 2, "pad": "SAME"}),
                 ("conv", {"patch_x": 5, "patch_y": 5, "depth": 64, "channels": 32}),
                 ("pool", {"side": 2, "stride": 2, "pad": "SAME"})]

    seed = 100
    C = Recogniser_Type1(structure=structure, nb_classes=10, img_rows=28, img_cols=28, max_nb_digits=0)

    optim = {"type": "Adadelta", "learning_rate": 0.01}
    C.fit(X, y, X_valid, y_valid, batch_size=128, nb_epochs=3500, logging_info=100, seed=seed, p_dropout=0.5, optimizer=optim)


def SVHN_single():
    import SVHN_data
    data = SVHN_data.SVHN_Single_Digit()
    X, y, X_valid, y_valid, X_test, y_test = data.datasets(format="digits")

    yc = np.ones((len(y), 2), dtype=np.int64)
    yc_valid = np.ones((len(y_valid), 2), dtype=np.int64)
    yc_test = np.ones((len(y_test), 2), dtype=np.int64)
    yc[:, 1] = y
    yc_valid[:, 1] = y_valid
    yc_test[:, 1] = y_test

    print yc.shape, yc_valid.shape, yc_test.shape
    print X.shape, X_valid.shape, X_test.shape

    exit()
    structure1 = [("conv", {"patch_x": 5, "patch_y": 5, "depth": 32, "channels": 1}),
                  ("pool", {"side": 2, "stride": 2, "pad": "SAME"}),
                  ("conv", {"patch_x": 5, "patch_y": 5, "depth": 64, "channels": 32}),
                  ("pool", {"side": 2, "stride": 2, "pad": "SAME"})]

    model = Recogniser_Type2(structure=structure1, nb_classes=10, img_rows=32, img_cols=32, max_nb_digits=1, nb_hidden=512)
    n_e = 100000
    logging_info = 200
    seed = 100

    optim = {"type" :"Adagrad", "learning_rate": 0.01}
    model.fit(X, yc, X_valid, yc_valid, X_test, yc_test, batch_size=128, nb_epochs=n_e,
               logging_info=logging_info, seed=seed, p_dropout=0.5, optimizer=optim)

if __name__ == "__main__":
    #MNIST_SD()
    #exit()

    SVHN_single()

