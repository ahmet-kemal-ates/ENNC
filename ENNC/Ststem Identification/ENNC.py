import os
import sys
import numpy as np
import tensorflow as tf

# Use tf.keras to match your TF install (fixes "could not be resolved" in IDE)
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, RNN, Add, Multiply, Concatenate, Lambda
from tensorflow.keras import backend as K

# Make local libs importable: ../../Libraries/Python
HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "Libraries", "Python"))
if LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)

from customLayers import FunctionalLink
from circuitRNNcells import VdynState


# --- Small, serializable layer to clamp τ into (minTau, maxTau) ---
class TauBound(tf.keras.layers.Layer):
    def __init__(self, min_tau, max_tau, **kwargs):
        super().__init__(**kwargs)
        self.min_tau = float(min_tau)
        self.max_tau = float(max_tau)

    def call(self, x):
        return self.min_tau + (self.max_tau - self.min_tau) * tf.math.sigmoid(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"min_tau": self.min_tau, "max_tau": self.max_tau})
        return cfg


class ENNC:
    def __init__(self, Cn, Ts, cRate_in, SoC_in, Temp_in,
                 num_neurons_Ist=25, num_hidden_Ist=3,
                 hiddenActivation_Ist='relu', outputActivation_Ist='linear',
                 num_tau=3, num_neurons_Dyn=25, num_hidden_Dyn=3, maxTau=5000, minTau=5,
                 hiddenActivation_Dyn='relu', outputActivation_Dyn='linear',
                 num_cheby=0, num_bernstein=15, num_trig=8, num_bspline=0,
                 outputActivation_Qst='linear'):

        # General parameters
        self.Cn = Cn
        self.Ts = Ts
        self.cRate_in = cRate_in
        self.SoC_in = SoC_in
        self.Temp_in = Temp_in

        # Vist net
        self.num_neurons_Ist = num_neurons_Ist
        self.num_hidden_Ist = num_hidden_Ist
        self.hiddenActivation_Ist = hiddenActivation_Ist
        self.outputActivation_Ist = outputActivation_Ist

        # Vdyn net
        self.num_tau = num_tau
        self.maxTau = maxTau
        self.minTau = minTau
        self.num_neurons_Dyn = num_neurons_Dyn
        self.num_hidden_Dyn = num_hidden_Dyn
        self.hiddenActivation_Dyn = hiddenActivation_Dyn
        self.outputActivation_Dyn = outputActivation_Dyn

        # Vqst net
        self.num_cheby = num_cheby
        self.num_bernstein = num_bernstein
        self.num_trig = num_trig
        self.num_bspline = num_bspline
        self.outputActivation_Qst = outputActivation_Qst

        ### BUILD NETWORK ###
        Iin  = Input(shape=(None, 1), name='Iin')
        SoC  = Input(shape=(None, 1), name='SoC')
        Temp = Input(shape=(None, 1), name='Temp')

        # Build parametric input
        if Temp_in:
            if cRate_in:
                componentInput = Concatenate()([Iin, Temp, SoC]) if SoC_in else Concatenate()([Iin, Temp])
            else:
                componentInput = Concatenate()([Temp, SoC]) if SoC_in else Temp
        else:
            if cRate_in:
                componentInput = Concatenate()([Iin, SoC]) if SoC_in else Iin
            else:
                componentInput = SoC if SoC_in else Lambda(lambda x: x**0)(Iin)

        # Initialize per-branch inputs
        Rist   = componentInput
        Rdyn   = componentInput
        tauDyn = componentInput
        Vqst   = Concatenate()([SoC, Temp]) if Temp_in else SoC

        # --- Rist (instantaneous) ---
        for n in range(num_hidden_Ist):
            Rist = TimeDistributed(
                Dense(num_neurons_Ist, activation=hiddenActivation_Ist, kernel_initializer='he_normal'),
                name=f'HidRistNet_{n}'
            )(Rist)
        Rist = TimeDistributed(
            Dense(1, activation=outputActivation_Ist, kernel_initializer='he_normal'),
            name='OutRistNet'
        )(Rist)
        Vist = Multiply(name='OutIstNet')([Rist, Iin])

        # --- Dynamic branch (Rdyn * Iin and τ) ---
        Iin_p = Iin
        for _ in range(num_tau - 1):
            Iin_p = Concatenate()([Iin_p, Iin])

        for n in range(num_hidden_Dyn):
            Rdyn = TimeDistributed(
                Dense(num_neurons_Dyn, activation=hiddenActivation_Dyn, kernel_initializer='he_normal'),
                name=f'HidRdynNet_{n}'
            )(Rdyn)
        Rdyn = TimeDistributed(
            Dense(num_tau, activation=outputActivation_Dyn, kernel_initializer='he_normal'),
            name='OutRdynNet'
        )(Rdyn)
        RdynI = Multiply()([Rdyn, Iin_p])

        # τ head
        for n in range(num_hidden_Dyn):
            tauDyn = TimeDistributed(
                Dense(num_neurons_Dyn, activation=hiddenActivation_Dyn, kernel_initializer='he_normal'),
                name=f'HidTauDynNet_{n}'
            )(tauDyn)
        tauDyn = TimeDistributed(
            Dense(num_tau, activation='linear', kernel_initializer='he_normal'),
            name='OutTauDynNet'
        )(tauDyn)
        # Clamp τ to a safe positive range
        tauDyn = TauBound(min_tau=minTau, max_tau=maxTau, name='TauBound')(tauDyn)

        # Concatenate τ and Rdyn*Iin for the recurrent cell
        Vdyn_in = Concatenate()([tauDyn, RdynI])

        # Recurrent dynamic layer
        cell_dyn = VdynState(num_tau, Ts=Ts, maxTau=maxTau, minTau=minTau)
        Vdyn = RNN(cell_dyn, return_sequences=True, name='VdynState')(Vdyn_in)

        # Mix all Vdyn branches
        trainable = self.num_tau != 1
        init = 'glorot_normal' if trainable else 'ones'  # 'ones' when only 1 τ
        Vdyn = TimeDistributed(
            Dense(1, activation='linear', kernel_initializer=init, use_bias=False),
            trainable=trainable, name='OutDynNet'
        )(Vdyn)

        # --- Quasi-static branch ---
        Vqst = TimeDistributed(
            FunctionalLink(num_cheby=num_cheby, num_trig=num_trig, num_bernstein=num_bernstein, num_bspline=num_bspline),
            name='HidQstNet'
        )(Vqst)
        Vqst = TimeDistributed(
            Dense(1, activation=outputActivation_Qst, kernel_initializer='he_normal'),
            name='OutQstNet'
        )(Vqst)

        # Output
        Vout = Add()([Vist, Vqst, Vdyn])

        # Final model + handy sub-models
        if Temp_in:
            self.net = Model(inputs=[Iin, SoC, Temp], outputs=Vout)
            self.VistFnc   = Model(self.net.inputs, self.net.get_layer('OutIstNet').output)
            self.VdynFnc   = Model(self.net.inputs, self.net.get_layer('OutDynNet').output)
            self.VdynsFnc  = Model(self.net.inputs, self.net.get_layer('OutDynNet').input)
            self.VqstFnc   = Model(self.net.inputs, self.net.get_layer('OutQstNet').output)
            self.RistFnc   = Model(self.net.inputs, self.net.get_layer('OutRistNet').output)
            self.RdynFnc   = Model(self.net.inputs, self.net.get_layer('OutRdynNet').output)
            self.TauDynFnc = Model(self.net.inputs, self.net.get_layer('OutTauDynNet').output)  # raw τ head
        else:
            self.net = Model(inputs=[Iin, SoC], outputs=Vout)
            self.VistFnc   = Model(self.net.inputs, self.net.get_layer('OutIstNet').output)
            self.VdynFnc   = Model(self.net.inputs, self.net.get_layer('OutDynNet').output)
            self.VdynsFnc  = Model(self.net.inputs, self.net.get_layer('OutDynNet').input)
            self.VqstFnc   = Model(self.net.inputs, self.net.get_layer('OutQstNet').output)
            self.RistFnc   = Model(self.net.inputs, self.net.get_layer('OutRistNet').output)
            self.RdynFnc   = Model(self.net.inputs, self.net.get_layer('OutRdynNet').output)
            self.TauDynFnc = Model(self.net.inputs, self.net.get_layer('OutTauDynNet').output)

    # Optional helper; main script uses net.compile/fit directly
    def fit(self, x_tr, y_tr, nEpoch=2000, batchSize=1, optimizer='Nadam', loss='mse'):
        if not hasattr(self, "_compiled") or not self._compiled:
            opt = optimizer if isinstance(optimizer, tf.keras.optimizers.Optimizer) else tf.keras.optimizers.get(optimizer)
            self.net.compile(optimizer=opt, loss=loss)
            self._compiled = True
        history = self.net.fit(x_tr, y_tr, epochs=nEpoch, batch_size=batchSize, verbose=2)
        return history

    def GetWeights(self):
        Rist_hidden = []
        for n in range(self.num_hidden_Ist):
            Rist_hidden.append(self.net.get_layer(name=f'HidRistNet_{n}').get_weights())
        Rist_out = self.net.get_layer(name='OutRistNet').get_weights()

        Rdyn_hidden = []
        for n in range(self.num_hidden_Dyn):
            Rdyn_hidden.append(self.net.get_layer(name=f'HidRdynNet_{n}').get_weights())
        Rdyn_out = self.net.get_layer(name='OutRdynNet').get_weights()

        tauDyn_hidden = []
        for n in range(self.num_hidden_Dyn):
            tauDyn_hidden.append(self.net.get_layer(name=f'HidTauDynNet_{n}').get_weights())
        tauDyn_out = self.net.get_layer(name='OutTauDynNet').get_weights()
        tauDyn_gain = self.net.get_layer(name='VdynState').get_weights()

        Vdyn_out = self.net.get_layer(name='OutDynNet').get_weights()
        Vqst_out = self.net.get_layer(name='OutQstNet').get_weights()

        if self.Temp_in:
            if self.cRate_in and not self.SoC_in:
                Rist_hidden[0][0] = np.concatenate((Rist_hidden[0][0], np.zeros((1, Rist_hidden[0][0].shape[1]))), axis=0)
                Rdyn_hidden[0][0] = np.concatenate((Rdyn_hidden[0][0], np.zeros((1, Rdyn_hidden[0][0].shape[1]))), axis=0)
                tauDyn_hidden[0][0] = np.concatenate((tauDyn_hidden[0][0], np.zeros((1, tauDyn_hidden[0][0].shape[1]))), axis=0)
            elif (not self.cRate_in) and self.SoC_in:
                Rist_hidden[0][0] = np.concatenate((np.zeros((1, Rist_hidden[0][0].shape[1])), Rist_hidden[0][0]), axis=0)
                Rdyn_hidden[0][0] = np.concatenate((np.zeros((1, Rdyn_hidden[0][0].shape[1])), Rdyn_hidden[0][0]), axis=0)
                tauDyn_hidden[0][0] = np.concatenate((np.zeros((1, tauDyn_hidden[0][0].shape[1])), tauDyn_hidden[0][0]), axis=0)
            elif (not self.cRate_in) and (not self.SoC_in):
                Rist_hidden[0][0] = np.concatenate((np.zeros((1, Rist_hidden[0][0].shape[1])), Rist_hidden[0][0], np.zeros((1, Rist_hidden[0][0].shape[1]))), axis=0)
                Rdyn_hidden[0][0] = np.concatenate((np.zeros((1, Rdyn_hidden[0][0].shape[1])), Rdyn_hidden[0][0], np.zeros((1, Rdyn_hidden[0][0].shape[1]))), axis=0)
                tauDyn_hidden[0][0] = np.concatenate((np.zeros((1, tauDyn_hidden[0][0].shape[1])), tauDyn_hidden[0][0], np.zeros((1, tauDyn_hidden[0][0].shape[1]))), axis=0)
        else:
            if self.cRate_in and (not self.SoC_in):
                Rist_hidden[0][0] = np.concatenate((Rist_hidden[0][0], np.zeros((1, Rist_hidden[0][0].shape[1]))), axis=0)
                Rdyn_hidden[0][0] = np.concatenate((Rdyn_hidden[0][0], np.zeros((1, Rdyn_hidden[0][0].shape[1]))), axis=0)
                tauDyn_hidden[0][0] = np.concatenate((tauDyn_hidden[0][0], np.zeros((1, tauDyn_hidden[0][0].shape[1]))), axis=0)
            elif (not self.cRate_in) and self.SoC_in:
                Rist_hidden[0][0] = np.concatenate((np.zeros((1, Rist_hidden[0][0].shape[1])), Rist_hidden[0][0]), axis=0)
                Rdyn_hidden[0][0] = np.concatenate((np.zeros((1, Rdyn_hidden[0][0].shape[1])), Rdyn_hidden[0][0]), axis=0)
                tauDyn_hidden[0][0] = np.concatenate((np.zeros((1, tauDyn_hidden[0][0].shape[1])), tauDyn_hidden[0][0]), axis=0)

        Rist_w = {'hiddenActivation': self.hiddenActivation_Ist, 'outputActivation': self.outputActivation_Ist,
                  'W_i2h': Rist_hidden, 'W_h2o': Rist_out}
        Rdyn_w = {'hiddenActivation': self.hiddenActivation_Dyn, 'outputActivation': self.outputActivation_Dyn,
                  'W_i2h': Rdyn_hidden, 'W_h2o': Rdyn_out}
        tauDyn_w = {'hiddenActivation': self.hiddenActivation_Dyn, 'outputActivation': 'sigmoid',
                    'W_i2h': tauDyn_hidden, 'W_h2o': tauDyn_out, 'gain': tauDyn_gain,
                    'maxTau': float(self.maxTau), 'minTau': float(self.minTau)}
        Vdyn_w = {'W_mix': Vdyn_out}
        Vqst_w = {'outputActivation': self.outputActivation_Qst,
                  'num_cheby': float(self.num_cheby), 'num_trig': float(self.num_trig), 'num_bernstein': float(self.num_bernstein),
                  'W_h2o': Vqst_out}

        self.netWeights = {'Cn': self.Cn, 'Ts': float(self.Ts),
                           'Rist_w': Rist_w, 'Rdyn_w': Rdyn_w, 'tauDyn_w': tauDyn_w,
                           'Vdyn_w': Vdyn_w, 'Vqst_w': Vqst_w}
        return self.netWeights

    def SetTrainableIst(self, trainable):
        for n in range(self.num_hidden_Ist):
            self.net.get_layer(name=f'HidRistNet_{n}').trainable = trainable
        self.net.get_layer(name='OutRistNet').trainable = trainable

    def SetTrainableDyn(self, trainable):
        for n in range(self.num_hidden_Dyn):
            self.net.get_layer(name=f'HidRdynNet_{n}').trainable = trainable
        self.net.get_layer(name='OutRdynNet').trainable = trainable
        for n in range(self.num_hidden_Dyn):
            self.net.get_layer(name=f'HidTauDynNet_{n}').trainable = trainable
        self.net.get_layer(name='OutTauDynNet').trainable = trainable
        self.net.get_layer(name='VdynState').trainable = trainable
        self.net.get_layer(name='OutDynNet').trainable = trainable

    def SetTrainableQst(self, trainable):
        self.net.get_layer(name='OutQstNet').trainable = trainable