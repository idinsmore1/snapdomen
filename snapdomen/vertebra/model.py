from keras.layers import Layer, InputSpec
import keras.backend as K
from tensorflow.keras.models import model_from_json


class _GlobalHorizontalPooling2D(Layer):
    """Abstract class for different global pooling 2D layers.
    """
    def __init__(self, data_format=None, **kwargs):
        super(_GlobalHorizontalPooling2D, self).__init__(**kwargs)
        self.data_format = data_format
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        #     return (input_shape[0], input_shape[1], input_shape[2])
        # else:
        return input_shape[0], input_shape[1], input_shape[3]

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(_GlobalHorizontalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalMaxHorizontalPooling2D(_GlobalHorizontalPooling2D):
    """Global max pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        # if self.data_format == 'channels_last':
        return K.max(inputs, axis=[2])
        # else:
        #     return K.max(inputs, axis=[3])


def load_model(model_path):
    """
    Load a model from a file.
    :param model_path: the path to the model file
    :return: the model
    """
    with open(model_path, 'rb') as f:
        model_string = f.read()
    model = model_from_json(model_string, custom_objects={'GlobalMaxHorizontalPooling2D': GlobalMaxHorizontalPooling2D})
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model
