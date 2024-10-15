import tensorflow as tf
from tensorflow import keras
from keras import layers

class Net(tf.keras.Model):
    def __init__(self, cond_lambd, lambda_lim, dim, I, nbNeurons, activation, is_decoupled):
        super().__init__()
        self.nbNeurons = nbNeurons
        self.dim = dim
        self.I = I
        self.lambda_lim = lambda_lim
        self.activation = activation
        self.ListOfDense = [
            layers.Dense(nbNeurons[i], activation=activation, kernel_initializer=tf.keras.initializers.GlorotNormal())
            for i in range(len(nbNeurons))
        ] + [layers.Dense(self.dim * self.I, activation=None, kernel_initializer=tf.keras.initializers.GlorotNormal())]
        self.cond_lambd = cond_lambd

        if cond_lambd:
            lambda_lim = tf.constant(lambda_lim, dtype=tf.float32)
            if is_decoupled:
                initial_value_shape = (I,)
            else:
                initial_value_shape = (1,)

            initial_value = tf.constant(0.0, shape=initial_value_shape, dtype=tf.float32)
            initial_value = tf.clip_by_value(initial_value, -lambda_lim, lambda_lim)
            self.lambd = tf.Variable(
                initial_value,
                trainable=True,
                dtype=tf.float32,
                constraint=lambda t: tf.clip_by_value(t, -lambda_lim, lambda_lim),
                name='lambd'
            )


    def call(self, cond_lambd, inputs):
        x = inputs
        for layer in self.ListOfDense:
            x = layer(x)

        if self.dim == 1:
            x = x[:, tf.newaxis, :]
        else:
            x = tf.reshape(x, [-1, self.dim, self.I])

        return x