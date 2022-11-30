from abc import ABC, abstractmethod
from tensorflow.python.keras.layers import Layer, Flatten, Rescaling
from tensorflow_hub import KerasLayer


class TransferLearningModel(ABC, Layer):
    IMAGE_SIZE = None

    @staticmethod
    @abstractmethod
    def find_base_model(model):
        """
        Locates the base model layer in an existing compiled model.
        """

    @abstractmethod
    def unfreeze(self, layer_breakpoint: str = None):
        """
        Unfreezes the model from the top layer down to the specified layer breakpoint.
        :param layer_breakpoint: the name of the layer above which all layers are unfrozen
        """


class ApplicationBasedTransferLearningModel(TransferLearningModel, ABC):
    key = ''

    @staticmethod
    @abstractmethod
    def _create_base_model(*args, **kwargs):
        """
        Create a base model with the given parameters.
        Necessary to work around self-parameter binding.
        """

    @staticmethod
    @abstractmethod
    def _create_preprocessor(*args, **kwargs):
        """
        Create a preprocessor with the given parameters.
        Necessary to work around self-parameter binding.
        """

    @staticmethod
    def _index_base_model_layer_by_name(model, name):
        return model.layers.index(next((layer for layer in model.layers
            if layer.name == name), None))

    @staticmethod
    def find_base_model(model):
        return next((layer for layer in model.layers
            if layer.name == cls.key), None)

    def __init__(self, *args, factory=None, preprocessor=None, \
                 input_shape=None, trainable=True, **kwargs):
        super().__init__()
        self._factory = factory or self._create_base_model
        self._preprocessor = preprocessor or self._create_preprocessor()
        self._base_model = self._factory(
            input_shape=input_shape,
            include_top=False,
            *args, **kwargs
        )
        self._base_model.trainable = trainable

    def call(self, inputs, *args, **kwargs):
        x = self._preprocessor(inputs)
        x = self._base_model(inputs, *args, **kwargs)
        x = Flatten()(x)
        return x

    def unfreeze(self, layer_breakpoint: str):
        layer_breakpoint_index = self._index_base_model_layer_by_name(self._base_model, name=layer_breakpoint)
        self._base_model.trainable = True
        for layer in self._base_model.layers[:layer_breakpoint_index+1]:
            layer.trainable = False


class LayerBasedTransferLearningModel(TransferLearningModel, ABC):
    handle = None

    @classmethod
    def find_base_model(cls, model):
        # HACK: obviously won't work if multiple keras layers are used
        return next((layer for layer in model.layers
            if isinstance(layer, cls)), None)

    def __init__(self, *args, handle=None, input_shape=None, trainable=True, **kwargs):
        super().__init__()
        self._base_model = KerasLayer(
            handle or self.handle,
            input_shape=input_shape,
            trainable=trainable,
        )

    def call(self, inputs, *args, **kwargs):
        x = Rescaling(1./255)(inputs)
        x = self._base_model(inputs, *args, **kwargs)
        return x

    def unfreeze(self, layer_breakpoint: str = None):
        self._base_model.trainable = True
        self._base_model.arguments = dict(
            batch_norm_momentum=0.997
        )
