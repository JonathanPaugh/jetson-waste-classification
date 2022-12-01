from tensorflow.python.keras.applications.inception_v3 import InceptionV3 as _InceptionV3, \
    preprocess_input
from core.transfer_learning import ApplicationBasedTransferLearningModel


class InceptionV3(ApplicationBasedTransferLearningModel):

    @staticmethod
    def _create_base_model(*args, **kwargs):
        return _InceptionV3(*args, **kwargs)

    @staticmethod
    def _create_preprocessor():
        return preprocess_input
