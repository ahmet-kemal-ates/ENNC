# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import math
import copy
import types as python_types
import warnings

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import Layer
from keras.layers import InputSpec


import scipy.special

class WNN(Layer):
    """ Densely-connected NN layer with Wavelet Neurons.

    `Wavelet` implements the operation:
    `output = prod(wavelet((input-T)/D)`
    where `wavelet` is the element-wise wavelet function,
    `T` is the delay term and 'D' is the dilatation coefficient.

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Wavelet(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Wavelet(32))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        wavelet: name of the wavelet to be used between 'morlet' and 'mexican'.
            If you don't specify anything, the morlet wavelet is used.

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """


    def __init__(self, levels=0,
                 wavelet='morlet',
                 initialize_range=[0.0, 1.0],
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(WNN, self).__init__(**kwargs)
        self.levels = levels
        self.units = int(sum(2**(np.linspace(0,levels,levels+1))))
        self.wavelet = wavelet
        self.initialize_range = initialize_range
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]


        self.T = self.add_weight((input_dim, self.units),
                                 initializer='zero',
                                 name='translation')
        self.D = self.add_weight((input_dim, self.units),
                                 initializer='zero',
                                 name='dilatation')

        translation,dilatation = self.tree_builder(self.initialize_range[0], self.initialize_range[1], 0)
        for L in range(1,self.levels+1):
            t, d = self.tree_builder(self.initialize_range[0], self.initialize_range[1], L)
            translation = np.concatenate((translation, t), axis=0)
            dilatation = np.concatenate((dilatation, d), axis=0)

        self.set_weights([translation.reshape((input_dim, self.units)), dilatation.reshape((input_dim, self.units))])

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        inputWavelet = inputs[:, :, None]
        inputWavelet = (tf.repeat(inputWavelet, self.units, axis=2))

        if self.wavelet == 'morlet':
            out = self.morlet((inputWavelet-self.T)/self.D)
        elif self.wavelet == 'mexican':
            out = self.mexican((inputWavelet-self.T)/self.D)

        return tf.reduce_prod(out, axis=1)

    def tree_builder(self, a, b, L):
        traslation = []
        dilatation = []
        if L == 0:
            traslation = np.array([(a+b)/2]).reshape(1,1)
            dilatation = np.array([b-a]).reshape(1,1)
            return traslation, dilatation
        else:
            t, d = self.tree_builder(a, np.mean([b,a]), L - 1)
            traslation.append(t)
            dilatation.append(d)

            t,d = self.tree_builder(np.mean([b,a]), b, L - 1)
            traslation.append(t)
            dilatation.append(d)

        return np.array(traslation).flatten()[:,None], np.array(dilatation).flatten()[:,None]

    @staticmethod
    def morlet(x):
        y = tf.cos(1.75 * x) * tf.exp(-0.5 * x ** 2)
        return y

    @staticmethod
    def mexican(x):
        y = tf.exp(-x ** 2) * (1 - x) ** 2
        return y

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'wavelet': self.wavelet
        }
        base_config = super(WNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Wavenet(Layer):
    """ Densely-connected NN layer with Wavelet Neurons.

    `Wavelet` implements the operation:
    `output = prod(wavelet((input-T)/D)`
    where `wavelet` is the element-wise wavelet function,
    `T` is the delay term and 'D' is the dilatation coefficient.

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Wavelet(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Wavelet(32))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        wavelet: name of the wavelet to be used between 'morlet' and 'mexican'.
            If you don't specify anything, the morlet wavelet is used.

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """


    def __init__(self, levels=0,
                 initialize_range=[0.0, 1.0],
                 trainable=False,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Wavenet, self).__init__(trainable=trainable,**kwargs)
        self.levels = levels
        self.units = int(2 ** levels)
        self.initialize_range = initialize_range
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]


        self.T = self.add_weight((input_dim, self.units),
                                 initializer='zero',
                                 name='translation')
        self.D = self.add_weight((input_dim, self.units),
                                 initializer='zero',
                                 name='dilatation')

        translation,dilatation = self.tree_builder(self.initialize_range[0], self.initialize_range[1], self.levels)

        self.set_weights([translation.reshape((input_dim, self.units)), dilatation.reshape((input_dim, self.units))])

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        inputWavelet = inputs[:, :, None]
        inputWavelet = (tf.repeat(inputWavelet, self.units, axis=2))

        out = self.meyer_scaling((inputWavelet - self.T)/self.D)

        return tf.reduce_prod(out, axis=1)

    def tree_builder(self, a, b, L):
        traslation = []
        dilatation = []
        if L == 0:
            traslation = np.array([(a+b)/2]).reshape(1,1)
            dilatation = np.array([(b-a)]).reshape(1,1)
            return traslation, dilatation
        else:
            t, d = self.tree_builder(a, np.mean([b,a]), L - 1)
            traslation.append(t)
            dilatation.append(d)

            t,d = self.tree_builder(np.mean([b,a]), b, L - 1)
            traslation.append(t)
            dilatation.append(d)

        return np.array(traslation).flatten()[:,None], np.array(dilatation).flatten()[:,None]

    @staticmethod
    def meyer_scaling(x):
        x += 1e-16
        y = (tf.sin(2/3*np.pi*x) + 4/3*x*tf.cos(4/3*np.pi*x)) / (np.pi*x - 16/9*np.pi*x**3)
        return y

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'levels': self.levels,
            'units': self.units
        }
        base_config = super(Wavenet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Wavelet(Layer):
    """ Densely-connected NN layer with Wavelet Neurons.

    `Wavelet` implements the operation:
    `output = prod(wavelet((input-T)/D)`
    where `wavelet` is the element-wise wavelet function,
    `T` is the delay term and 'D' is the dilatation coefficient.

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Wavelet(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Wavelet(32))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        wavelet: name of the wavelet to be used between 'morlet' and 'mexican'.
            If you don't specify anything, the morlet wavelet is used.

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """


    def __init__(self, units,
                 wavelet='morlet',
                 translation_initializer = 'ones',
                 dilatation_initializer = 'ones',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Wavelet, self).__init__(**kwargs)
        self.units = units
        self.wavelet = wavelet
        self.translation_initializer = translation_initializer
        self.dilatation_initializer = dilatation_initializer
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.T = self.add_weight((input_dim, self.units),
                                 initializer=self.translation_initializer,
                                 name='translation')
        self.D = self.add_weight((input_dim, self.units),
                                 initializer=self.dilatation_initializer,
                                 name='dilatation')

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        inputWavelet = inputs[:, :, None]
        inputWavelet = (tf.repeat(inputWavelet, (self.units), axis=2) - self.T) / self.D

        if self.wavelet == 'morlet':
            out = self.morlet(inputWavelet)
        elif self.wavelet == 'mexican':
            out = self.mexican(inputWavelet)

        return tf.reduce_prod(out, axis=1)

    @staticmethod
    def morlet(x):
        y = tf.cos(1.75 * x) * tf.exp(-0.5 * x ** 2)
        return y

    @staticmethod
    def mexican(x):
        y = tf.exp(-x ** 2) * (1 - x) ** 2
        return y

    def initialize_wavelons(self, translation, dilatation):
        dilatation = np.reshape(dilatation, (self.input_shape[-1], self.units))
        translation = np.reshape(translation, (self.input_shape[-1], self.units))
        self.set_weights([translation, dilatation])

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'wavelet': self.wavelet
        }
        base_config = super(Wavelet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RBF(Layer):
    """ Densely-connected Radial Basis Function layer.

    `RBF` implements the operation:
    `output = sum(exp(-B(input-U)**2)`
    where `B` is the dilatation factor and 'U' is the center of the RBF.

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(RBF(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(RBF(32))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """


    def __init__(self, units,
                 center_initializer='glorot_uniform',
                 std_initializer='ones',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(RBF, self).__init__(**kwargs)
        self.units = units
        self.center_initializer = center_initializer
        self.std_initializer = std_initializer
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.U = self.add_weight((input_dim, self.units),
                                 initializer=self.center_initializer,
                                 name='mean')
        self.B = self.add_weight((self.units,),
                                 initializer=self.std_initializer,
                                 name='sigma')

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        inputRBF = inputs[:, :, None]
        inputRBF = tf.repeat(inputRBF, (self.units), axis=2) - self.U
        return tf.exp(-self.B * tf.reduce_sum(inputRBF ** 2, axis=1) / self.units)

    def set_means(self, mean_values):
        #assert mean_values.shape == (self.input_dim, self.units)
        mean_values = np.reshape(mean_values, (self.input_shape[-1], self.units))
        self.set_weights([mean_values, self.B.eval()])

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'center_initializer': self.center_initializer,
            'std_initializer': self.std_initializer
        }
        base_config = super(RBF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FunctionalLink(Layer):
    """ Densely-connected Functional Link layer.

    `FunctionalLink` implements the operation:
    `output = concatenate(f1(input), f2(input), ... , fn(input))`
    where `fi` is the i-th function of the functional reservoir

    Three kind of functional reservoir are implemented: chebichev trigonometric
    and legendre polynomials.

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(FunctionalLink(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(FunctionalLink(32))
    ```

    # Arguments
        num_cheby: number of chebichev polynomials to be considered with grade from 0 to num_cheby.
        num_trig: number of trigonometric polynomials to be considered with grade from 0 to num_trig.
        num_legendre: number of legendre polynomials to be considered with grade from 0 to num_legendre.

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """


    def __init__(self, num_cheby=10, num_trig=0, num_bernstein=0, num_bspline=0, k=2,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FunctionalLink, self).__init__(**kwargs)
        self.num_cheby = num_cheby
        self.num_trig = num_trig
        self.num_bernstein = num_bernstein
        self.num_bspline = num_bspline
        self.k = k
        maxBinom = np.maximum(self.num_bernstein, self.num_cheby) + 1
        self.binom = np.zeros((maxBinom, maxBinom), dtype=tf.keras.backend.floatx())
        for i in range(maxBinom):
            for j in range(maxBinom):
                self.binom[i,j] = scipy.special.binom(i,j)

        self.units = num_cheby*(num_cheby != 0) \
            + (2*num_trig)*(num_trig != 0) \
            + (1+num_bernstein)*(num_bernstein != 0) \
            + (1+num_bspline)*(num_bspline != 0)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.units *= input_dim
        self.knots = np.linspace(-0.1, 1.1, self.num_bspline + self.k + 2, dtype=tf.keras.backend.floatx())
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        output = None

        if self.num_cheby != 0:
            output = self.cheby(inputs, 1)
            for n in range(2, self.num_cheby + 1):
                output = tf.concat((output, self.cheby(inputs, n)), axis=1)

        if self.num_trig != 0:
            if output is None:
                output = tf.sin(1 * inputs)
                output = tf.concat((output, tf.cos(1 * inputs)), axis=1)
                for n in range(2, self.num_trig + 1):
                    output = tf.concat((output, tf.sin(n * inputs)), axis=1)
                    output = tf.concat((output, tf.cos(n * inputs)), axis=1)
            else:
                for n in range(1, self.num_trig + 1):
                    output = tf.concat((output, tf.sin(n * inputs)), axis=1)
                    output = tf.concat((output, tf.cos(n * inputs)), axis=1)

        if self.num_bernstein != 0:
            if output is None:
                output = self.bernstein(inputs, 0, self.num_bernstein)
                for n in range(1, self.num_bernstein + 1):
                    output = tf.concat((output, self.bernstein(inputs, n, self.num_bernstein)), axis=1)
            else:
                for n in range(0, self.num_bernstein + 1):
                    output = tf.concat((output, self.bernstein(inputs, n, self.num_bernstein)), axis=1)

        if self.num_bspline != 0:
            if output is None:
                output = self.bspline(inputs, 0, self.k)
                for n in range(1, self.num_bspline + 1):
                    output = tf.concat((output, self.bspline(inputs, n, self.k)), axis=1)
            else:
                for n in range(0, self.num_bspline + 1):
                    output = tf.concat((output, self.bspline(inputs, n, self.k)), axis=1)

        return output

    def cheby(self, x, n):
        N = math.floor(n / 2)
        y = x * 0
        for m in range(N + 1):
            y += self.binom[n, 2 * m] * (x ** (n - 2 * m)) * (x ** 2 - 1) ** m
        return y

    def bernstein(self, x, n, k):
        y = self.binom[k, n] * x**n * (1-x)**(k-n)
        return y

    def bspline(self, x, n, k, level=0, flag=False):
        if level==0 and (n==0 or n==self.num_bspline):
            flag = True

        if k == 0:
            #if flag and n == 0:
            #    condition = x < self.knots[n + 1]
            #elif flag and n == self.num_bspline:
            #    condition = (x >= self.knots[n])
            #else:
            condition = (x >= self.knots[n]) * (x < self.knots[n + 1])
            return (x**(1-condition))*condition

        else:
            if self.knots[n+k] == self.knots[n]:
                c1 = x*0
            else:
                c1 = (x-self.knots[n])/(self.knots[n+k]-self.knots[n])*self.bspline(x, n, k-1, level, flag)

            if self.knots[n+k+1] == self.knots[n+1]:
                c2 = x*0
            else:
                c2 = (self.knots[n+k+1]-x)/(self.knots[n+k+1]-self.knots[n+1])*self.bspline(x, n+1, k-1, level, flag)
            return c1+c2

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'num_cheby': self.num_cheby,
            'num_trig': self.num_trig,
            'num_bernstein': self.num_bernstein,
            'num_bspline': self.num_bspline
        }
        base_config = super(FunctionalLink, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


