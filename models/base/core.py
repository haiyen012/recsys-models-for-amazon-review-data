import tensorflow as tf

try:
    from tensorflow.python.ops.init_ops_v2 import Zeros
except ImportError:
    from tensorflow.python.ops.init_ops import Zeros

from tensorflow.python.keras.layers import Dense, Layer

try:
    from tensorflow.python.keras.layers import BatchNormalization
except ImportError:
    BatchNormalization = tf.keras.layers.BatchNormalization

from deepctr.layers.core import DNN


class PredictionLayer(Layer):
    def __init__(self, task="binary", num_classes=None, use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.num_classes = num_classes
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.task == "multiclass":
            if self.num_classes is None:
                raise ValueError(
                    "`num_classes` must be specified for 'multiclass' task."
                )
                # Dense layer to generate outputs for multi-class tasks
            self.dense = Dense(self.num_classes, use_bias=self.use_bias)

        if self.use_bias and self.task != "multiclass":
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias"
            )

            # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.task == "binary":
            x = tf.sigmoid(x)
            if self.use_bias:
                x = tf.nn.bias_add(x, self.global_bias, data_format="NHWC")
            output = tf.reshape(x, (-1, 1))
        elif self.task == "multiclass":
            x = self.dense(x)
            output = tf.nn.softmax(x)
        print(output)
        # output_shape = tf.shape(output)  # Get the shape as a NumPy array
        # print("Output shape:", output_shape)
        return output

    def compute_output_shape(self, input_shape):
        if self.task == "binary":
            return (input_shape[0], 1)  # Binary output has shape (batch_size, 1)
        elif self.task == "multiclass":
            return (
                input_shape[0],
                self.num_classes,
            )  # Multi-class output has shape (batch_size, num_classes)
        elif self.task == "regression":
            return (input_shape[0], 1)  # Regression output has shape (batch_size, 1)

    def get_config(
        self,
    ):
        config = {
            "task": self.task,
            "num_classes": self.num_classes,
            "use_bias": self.use_bias,
        }
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
