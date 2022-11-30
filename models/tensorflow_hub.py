"""
Details layer-based transfer learning models from TensorFlow Hub.
Note that using these models may require configuring specific image sizes in `configs/model.py`.
"""

from core.transfer_learning import LayerBasedTransferLearningModel


class InceptionV3(LayerBasedTransferLearningModel):
    handle = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5'

class MobileNetV2x224(LayerBasedTransferLearningModel):
    IMAGE_SIZE = (224, 224)
    handle = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5'

class MobileNetV2x192(LayerBasedTransferLearningModel):
    IMAGE_SIZE = (192, 192)
    handle = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/5'

class MobileNetV2x128(LayerBasedTransferLearningModel):
    IMAGE_SIZE = (128, 128)
    handle = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/5'

class MobileNetV2x96(LayerBasedTransferLearningModel):
    IMAGE_SIZE = (96, 96)
    handle = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5'
