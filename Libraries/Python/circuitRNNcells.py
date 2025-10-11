from keras.layers import Layer
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints


class VqstState(Layer):

    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):

        self.units = units

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.state_size = self.units
        super(VqstState, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.mask_x = K.concatenate((K.ones((input_dim-1,)), K.zeros((1,))), axis=0)
        self.mask_I = K.concatenate((K.zeros((input_dim-1,)), K.ones((1,))), axis=0)

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                        initializer=self.kernel_initializer,
                                        name='kernel')

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                          initializer=self.bias_initializer,
                                          name='bias')

        else:
            self.bias = None


        self.gain = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='gain')

        self.built = True

    def call(self, inputs, states):
        s = states[0]

        x = inputs*self.mask_x
        I = K.sum(inputs*self.mask_I)

        x = K.dot(inputs, self.kernel)
        x_s = K.dot(s, self.expanding_kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)
            x_s = K.bias_add(x_s, self.bias_s)

        r = K.dot(x_s, self.recurrent_kernel)
        if self.use_bias:
            r = K.bias_add(r, self.bias_r)
        r = self.recurrent_activation(r)

        h = self.activation(x + r)*self.gain

        output = s + h*I

        return output, [output]


    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer)
                  }
        base_config = super(VqstState, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class VqstStateGRU(Layer):

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 **kwargs):

        self.units = units

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.state_size = self.units
        super(VqstStateGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.mask_x = K.concatenate((K.ones((input_dim-1,)), K.zeros((1,))), axis=0)
        self.mask_I = K.concatenate((K.zeros((input_dim-1,)), K.ones((1,))), axis=0)

        self.kernel_r = self.add_weight(shape=(input_dim, self.units),
                                        initializer=self.kernel_initializer,
                                        name='kernel_r')
        self.recurrent_kernel_r = self.add_weight(shape=(self.units, self.units),
                                                  initializer=self.recurrent_initializer,
                                                  name='recurrent_kernel_r')

        self.kernel_h = self.add_weight(shape=(input_dim, self.units),
                                        initializer=self.kernel_initializer,
                                        name='kernel_h')
        self.recurrent_kernel_h = self.add_weight(shape=(self.units, self.units),
                                                  initializer=self.recurrent_initializer,
                                                  name='recurrent_kernel_h')

        if self.use_bias:
            self.bias_r = self.add_weight(shape=(self.units,),
                                          initializer=self.bias_initializer,
                                          name='bias_r')

            self.bias_h = self.add_weight(shape=(self.units,),
                                          initializer=self.bias_initializer,
                                          name='bias_h')
        else:
            self.bias_r = None
            self.bias_h = None


        self.gain = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      name='gain')

        self.built = True

    def call(self, inputs, states):
        s = states[0]

        x = inputs*self.mask_x
        I = K.sum(inputs*self.mask_I)

        x_r = K.dot(x, self.kernel_r)
        x_h = K.dot(x, self.kernel_h)
        if self.use_bias:
            x_r = K.bias_add(x_r, self.bias_r)
            x_h = K.bias_add(x_h, self.bias_h)

        r = self.recurrent_activation(x_r + K.dot(s, self.recurrent_kernel_r))
        h = self.activation(x_h + K.dot(s*r, self.recurrent_kernel_h))*self.gain

        output = s + h*I

        return output, [output]


    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer)
                  }
        base_config = super(VqstStateGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class VdynState(Layer):

    def __init__(self, units, Ts=1.0, minTau=100.0, maxTau=10000.0, use_gain=True, **kwargs):
        self.units = units

        self.minTau = minTau
        self.maxTau = maxTau

        self.Ts = Ts

        self.use_gain = use_gain

        self.state_size = self.units
        super(VdynState, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_tau = K.concatenate((K.eye(self.units), K.zeros((self.units, self.units))),
                                   axis=0)
        self.w_RI = K.concatenate((K.zeros((self.units, self.units)), K.eye(self.units)),
                                 axis=0)

        if self.use_gain:
            self.gain = self.add_weight((self.units,),
                                        initializer='ones',
                                        name='gain')
        else:
            self.gain = 1

        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        tau = (self.minTau + K.dot(inputs, self.w_tau) * (self.maxTau-self.minTau)) * self.gain
        alpha = K.exp(-self.Ts / tau)
        RI = K.dot(inputs, self.w_RI)
        output = prev_output * alpha + RI * (1 - alpha)
        return output, [output]

    def get_config(self):
        config = {'units': self.units,
                  'Ts': self.Ts,
                  'maxTau': self.maxTau,
                  'minTau': self.minTau
                  }
        base_config = super(VdynState, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VdynStateVanilla(Layer):

    def __init__(self, units, **kwargs):
        self.units = units

        self.state_size = self.units
        super(VdynStateVanilla, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_tau = K.concatenate((K.eye(self.units), K.zeros((self.units, self.units))), axis=0)
        self.w_RI = K.concatenate((K.zeros((self.units, self.units)), K.eye(self.units)), axis=0)

        self.gain = self.add_weight((self.units,),
                                    initializer='ones',
                                    name='gain')

        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        alpha = K.dot(inputs, self.w_tau) * self.gain
        RI = K.dot(inputs, self.w_RI)
        output = prev_output * alpha + RI * (1 - alpha)
        return output, [output]

    def get_config(self):
        config = {'units': self.units}
        base_config = super(VdynStateVanilla, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VdynStateGRU(Layer):

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 **kwargs):

        self.units = units

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.state_size = self.units
        super(VdynStateGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.mask_x = K.concatenate((K.ones((input_dim-1,)), K.zeros((1,))), axis=0)
        self.mask_I = K.concatenate((K.zeros((input_dim-1,)), K.ones((1,))), axis=0)

        self.kernel_z = self.add_weight(shape=(input_dim, self.units),
                                    initializer=self.kernel_initializer,
                                    name='kernel_z')
        self.recurrent_kernel_z = self.add_weight(shape=(self.units, self.units),
                                        initializer=self.recurrent_initializer,
                                        name='recurrent_kernel_z')

        self.kernel_r = self.add_weight(shape=(input_dim, self.units),
                                        initializer=self.kernel_initializer,
                                        name='kernel_r')
        self.recurrent_kernel_r = self.add_weight(shape=(self.units, self.units),
                                                  initializer=self.recurrent_initializer,
                                                  name='recurrent_kernel_r')

        self.kernel_h = self.add_weight(shape=(input_dim, self.units),
                                        initializer=self.kernel_initializer,
                                        name='kernel_h')
        self.recurrent_kernel_h = self.add_weight(shape=(self.units, self.units),
                                                  initializer=self.recurrent_initializer,
                                                  name='recurrent_kernel_h')


        if self.use_bias:
            self.bias_z = self.add_weight(shape=(self.units,),
                                          initializer=self.bias_initializer,
                                          name='bias_z')

            self.bias_r = self.add_weight(shape=(self.units,),
                                          initializer=self.bias_initializer,
                                          name='bias_r')

            self.bias_h = self.add_weight(shape=(self.units,),
                                          initializer=self.bias_initializer,
                                          name='bias_h')
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None

        self.built = True

    def call(self, inputs, states):
        s = states[0]

        x = inputs*self.mask_x
        I = K.sum(inputs*self.mask_I)

        x_z = K.dot(x, self.kernel_z)
        x_r = K.dot(x, self.kernel_r)
        x_h = K.dot(x, self.kernel_h)
        if self.use_bias:
            x_z = K.bias_add(x_z, self.bias_z)
            x_r = K.bias_add(x_r, self.bias_r)
            x_h = K.bias_add(x_h, self.bias_h)

        z = self.recurrent_activation(x_z + K.dot(s, self.recurrent_kernel_z))
        r = self.recurrent_activation(x_r + K.dot(s, self.recurrent_kernel_r))
        h = self.activation(x_h + K.dot(s*r, self.recurrent_kernel_h))*I

        output = s*z + h*(1 - z)

        return output, [output]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer)
                  }
        base_config = super(VdynStateGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))