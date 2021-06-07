import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

class MNIST(object):
    def __init__(self, features_path = "../data/MNIST/train-images.idx3-ubyte",
                 labels_path = "../data/MNIST/train-labels.idx1-ubyte"):
        import idx2numpy
        logger.info("Loading MNIST data")
        logger.info("Loading Images...")
        f = open(features_path, 'rb')
        self.images = idx2numpy.convert_from_file(f)

        logger.info("Loading Labels...")
        f = open(labels_path, 'rb')
        self.labels = idx2numpy.convert_from_file(f)




#data = MNIST()
#print data.labels