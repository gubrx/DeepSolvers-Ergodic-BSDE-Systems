import tensorflow as tf
#from ErgodicFactorModel import ErgodicFactorModel
import time
from tensorflow import keras
from keras import optimizers


class SolverBase:
    # mathModel          Math model
    # modelKeras         Keras model
    # lRate              Learning rate
    def __init__(self, ErgodicFactorModel, lRate):
        self.ErgodicFactorModel = ErgodicFactorModel
        self.StochasticFactor = ErgodicFactorModel.stochastic_factor
        self.lRate = lRate

class SolverGlobaleBSDE(SolverBase):
    def __init__(self, ErgodicFactorModel, modelKerasUZ , lRate):
        super().__init__(ErgodicFactorModel, lRate)
        self.modelKerasUZ = modelKerasUZ

    def train(self, batchSize, batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def optimizeBSDE(nbSimul):
            print('Tracing optimizeBSDE...')
            start_time = tf.timestamp()
            T_A, dW, V = self.StochasticFactor.sample(self.StochasticFactor.mu, self.StochasticFactor.kappa, nbSimul) #(nb time stpe, nb simul, dim)
            maxind = tf.reduce_max(T_A)

            Y_trans = self.ErgodicFactorModel.Y0 * tf.ones([nbSimul, self.ErgodicFactorModel.I], dtype=tf.float32)  # Y0 list of size I
            Y_next = tf.zeros([nbSimul, self.ErgodicFactorModel.I], dtype=tf.float32)

            Y_array = tf.TensorArray(dtype=tf.float32, size=int(nbSimul))

            penalty_loss = 0.0
            penalty_factor = 0.0
            penalty_mask = tf.ones([nbSimul], dtype=tf.bool)
            mask = tf.linalg.band_part(tf.ones((self.ErgodicFactorModel.I, self.ErgodicFactorModel.I)), 0, -1) - tf.eye(self.ErgodicFactorModel.I)
            num_valid_terms = tf.cast(tf.reduce_sum(mask), tf.int32)

            for iStep in tf.range(int(maxind)):
                step_start_time = tf.timestamp()

                input_tensor = tf.expand_dims(V[iStep, :], axis=-1) #(nbSimul, 1)
                Z = self.modelKerasUZ(True, input_tensor) # shape (nbSimul, dim, I)

                # Y_trans shape (nbSimul, I)
                # Y_trans[:, i:i+1] (nbSimul, 1)
                interaction_terms = [
                    tf.reduce_sum(
                    self.ErgodicFactorModel.Q[i, :] * (Y_trans - Y_trans[:, i:i+1]), axis=1) for i in range(self.ErgodicFactorModel.I)]
                interaction_terms = tf.stack(interaction_terms, axis=1)
                if tf.reduce_any(tf.math.is_nan(interaction_terms)):
                    nan_indices = tf.where(tf.math.is_nan(interaction_terms))
                    tf.print('NaN detected in interaction_terms at step', iStep)
                    tf.print('Nan indices in interaction_terms:', nan_indices)
                    tf.print('Corresponding Y_trans:', tf.gather_nd(Y_trans, nan_indices))

                Y_next = Y_trans - self.StochasticFactor.dt * self.ErgodicFactorModel.f_vectorized(self.ErgodicFactorModel.f, V[iStep, :], Z) + self.modelKerasUZ.lambd*self.StochasticFactor.dt \
                - interaction_terms * self.StochasticFactor.dt + tf.reduce_sum(Z * dW[iStep, :, tf.newaxis], axis=1)

                indices = tf.where(T_A == iStep+1)[:, 0]
                indices = tf.cast(indices, tf.int32)

                Y_trans_selected = tf.gather(Y_next, indices)
                Y_array = Y_array.scatter(indices, Y_trans_selected)
                Y_trans = Y_next

            Y = Y_array.stack()

            return tf.reduce_mean(tf.reduce_sum(tf.square(Y - self.ErgodicFactorModel.Y0), axis=1))

        @tf.function
        def trainOptNN(nbSimul, optimizer):
            print('Train tracing')
            with tf.GradientTape() as tape:
                objFunc_Z = optimizeBSDE(nbSimul)
            gradients = tape.gradient(objFunc_Z, self.modelKerasUZ.trainable_variables)
            #Ignore potential nan gradients
            non_nan_gradients = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in gradients]
            #Gradient clipping
            clipped_gradients = [tf.clip_by_norm(grad, 1.0) for grad in non_nan_gradients]

            optimizer.apply_gradients(zip(clipped_gradients, self.modelKerasUZ.trainable_variables))
            return objFunc_Z

        optimizerN = optimizers.Adam(learning_rate = self.lRate)

        self.listlambd = []
        self.lossList = []
        self.duration = 0
        self.listlambd.append(self.modelKerasUZ.lambd.numpy())
        self.lossList.append(optimizeBSDE(batchSizeVal))
        for iout in range(num_epochExt):
            start_time = time.time()
            for epoch in range(num_epoch):
                # one stochastic gradient descent step
                trainOptNN(batchSize, optimizerN)
            end_time = time.time()
            rtime = end_time-start_time
            self.duration += rtime
            objError_Yterm = optimizeBSDE(batchSizeVal)
            lambd = self.modelKerasUZ.lambd.numpy()

            print(" Error",objError_Yterm.numpy(),  " elapsed time %5.3f s" % self.duration, "lambda so far ",lambd, 'epoch', iout)
            self.listlambd.append(lambd)
            self.lossList.append(objError_Yterm)

        return self.listlambd, self.lossList

class SolverLocaleBSDE(SolverBase):
    def __init__(self, ErgodicFactorModel, modelKerasY, modelKerasZ, lRate):
        super().__init__(ErgodicFactorModel, lRate)
        self.modelKerasY = modelKerasY
        self.modelKerasZ = modelKerasZ

    def train(self, batchSize, batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def optimizeBSDE(nbSimul):
            T_A, dW, V = self.StochasticFactor.sample(self.StochasticFactor.mu, self.StochasticFactor.kappa, nbSimul)
            maxind = tf.cast(tf.reduce_max(T_A), tf.int32)

            # Initialize Y_trans with shape (nbSimul, 1, I)
            input_tensor = tf.expand_dims(V[0, :], axis=-1)
            Y_trans = tf.squeeze(self.modelKerasY(False, input_tensor), axis=1)

            loss_tensor = tf.TensorArray(dtype=tf.float32, size=maxind+1)
            loss_term = tf.zeros([nbSimul, self.ErgodicFactorModel.I], dtype=tf.float32)

            err_zero = tf.square(Y_trans - self.ErgodicFactorModel.Y0)
            loss_tensor = loss_tensor.write(0, err_zero)

            for iStep in tf.range(maxind):
                input_tensor = tf.expand_dims(V[iStep, :], axis=-1)
                Z = self.modelKerasZ(True, input_tensor)
                input_Ynext = tf.expand_dims(V[iStep+1, :], axis=-1)
                Y_next = tf.squeeze(self.modelKerasY(False, input_Ynext), axis=1)

                interaction_terms = [
                    tf.reduce_sum(
                        self.ErgodicFactorModel.Q[i, :] * (
                        tf.exp(Y_trans[:, :] - Y_trans[:, i:i+1]) - 1), axis=1)
                    for i in range(self.ErgodicFactorModel.I)
                ]
                interaction_terms = tf.stack(interaction_terms, axis=1)  # Stack along the I-axis

                toAdd = (self.StochasticFactor.dt * self.ErgodicFactorModel.f_vectorized(self.ErgodicFactorModel.f, V[iStep, :], Z)
                        - self.modelKerasZ.lambd * self.StochasticFactor.dt
                        + interaction_terms * self.StochasticFactor.dt
                        - tf.reduce_sum(Z * dW[iStep, :, tf.newaxis], axis=1))

                loss_term = loss_term + toAdd

                indices = tf.where(T_A > iStep+1)[:, 0]
                loss_term_selec = tf.gather(loss_term, indices)
                Y_selec = tf.gather(Y_next, indices)

                local_loss = tf.zeros([nbSimul, self.ErgodicFactorModel.I], dtype=tf.float32)
                updates_loss = tf.square(Y_selec + loss_term_selec - self.ErgodicFactorModel.Y0)

                update_indices = tf.expand_dims(indices, axis=1)
                local_loss = tf.tensor_scatter_nd_update(local_loss, update_indices, updates_loss)

                indices_tau = tf.where(T_A == iStep+1)[:, 0]
                loss_term_tau = tf.gather(loss_term, indices_tau)

                updates_loss_tau = tf.square(loss_term_tau)

                update_indices_tau = tf.expand_dims(indices_tau, axis=1)
                local_loss = tf.tensor_scatter_nd_update(local_loss, update_indices_tau, updates_loss_tau)

                loss_tensor = loss_tensor.write(iStep+1, local_loss)

                Y_trans = Y_next

            loss_tensor = loss_tensor.stack()
            final_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(loss_tensor, axis=2), axis = 0))

            return final_loss


        @tf.function
        def trainOptNN(nbSimul, optimizer, optimizerLambda):
            with tf.GradientTape(persistent=True) as tape:
                objFunc = optimizeBSDE(nbSimul)

            gradientsZ = tape.gradient(objFunc, self.modelKerasZ.trainable_variables)
            gradientsY = tape.gradient(objFunc, self.modelKerasY.trainable_variables)

            all_gradients = gradientsY + gradientsZ
            all_variables = self.modelKerasY.trainable_variables + self.modelKerasZ.trainable_variables

            # all_gradients = [tf.clip_by_norm(grad, 1.0) for grad in all_gradients]

            optimizer.apply_gradients(zip(all_gradients, all_variables))

            del tape 
            return objFunc

        optimizer = optimizers.Adam(learning_rate=self.lRate)
        optimizerLambda = optimizers.Adam(learning_rate=self.lRate)

        self.listlambd = []
        self.lossList = []
        self.duration = 0
        self.listlambd.append(self.modelKerasZ.lambd.numpy())
        self.lossList.append(optimizeBSDE(batchSizeVal))
        for iout in range(num_epochExt):
            start_time = time.time()
            for epoch in range(num_epoch):
                # one stochastic gradient descent step
                trainOptNN(batchSize, optimizer, optimizerLambda)
            end_time = time.time()
            rtime = end_time-start_time

            self.duration += rtime
            objError_Yterm = optimizeBSDE(batchSizeVal)
            lambd = self.modelKerasZ.lambd.numpy()
            print(" Error",objError_Yterm.numpy(),  " elapsed time %5.3f s" % self.duration, "lambda so far ",lambd, 'epoch', iout)
            self.listlambd.append(lambd)
            self.lossList.append(objError_Yterm)


        return self.listlambd, self.lossList

class SolverADLocaleBSDE(SolverBase):
    def __init__(self, ErgodicFactorModel, modelKerasY, lRate):
        super().__init__(ErgodicFactorModel, lRate)
        self.modelKerasY = modelKerasY

    def train(self, batchSize, batchSizeVal, num_epoch, num_epochExt):
        @tf.function
        def optimizeBSDE(nbSimul):
            T_A, dW, V = self.StochasticFactor.sample(self.StochasticFactor.mu, self.StochasticFactor.kappa, nbSimul)
            maxind = tf.cast(tf.reduce_max(T_A), tf.int32)

            # Initialize Y_trans with shape (nbSimul, 1, I)
            input_tensor = tf.expand_dims(V[0, :], axis=-1)
            Y_trans = tf.squeeze(self.modelKerasY(True, input_tensor), axis=1)

            loss_tensor = tf.TensorArray(dtype=tf.float32, size=maxind+1)
            loss_term = tf.zeros([nbSimul, self.ErgodicFactorModel.I], dtype=tf.float32)

            err_zero = tf.square(Y_trans - self.ErgodicFactorModel.Y0)
            loss_tensor = loss_tensor.write(0, err_zero)

            for iStep in tf.range(maxind):
                input_tensor = tf.expand_dims(V[iStep, :], axis=-1)

                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(input_tensor)
                    Y_trans = self.modelKerasY(True, input_tensor)
                    Y_trans_list = [Y_trans[:, :, i] for i in range(self.ErgodicFactorModel.I)]

                gradients_list = []
                for i in range(self.ErgodicFactorModel.I):
                    Z_i = tape.gradient(Y_trans_list[i], input_tensor) # Z_i will have shape (nbSimul, 1)
                    gradients_list.append(Z_i)

                Z = tf.stack(gradients_list, axis=2)
                Z = self.StochasticFactor.kappa * Z

                input_Ynext = tf.expand_dims(V[iStep+1, :], axis=-1)
                Y_next = tf.squeeze(self.modelKerasY(True, input_Ynext), axis=1)

                Y_trans = tf.squeeze(Y_trans, axis=1)

                interaction_terms = [
                    tf.reduce_sum(
                        self.ErgodicFactorModel.Q[i, :] *(
                        tf.exp(Y_trans[:, :] - Y_trans[:, i:i+1]) - 1), axis=1)
                    for i in range(self.ErgodicFactorModel.I)
                ]
                interaction_terms = tf.stack(interaction_terms, axis=1)  # Stack along the I-axis

                toAdd = (self.StochasticFactor.dt * self.ErgodicFactorModel.f_vectorized(self.ErgodicFactorModel.f, V[iStep, :], Z)
                        - self.modelKerasY.lambd * self.StochasticFactor.dt
                        + interaction_terms * self.StochasticFactor.dt
                        - tf.reduce_sum(Z * dW[iStep, :, tf.newaxis], axis=1))

                loss_term = loss_term + toAdd

                indices = tf.where(T_A > iStep+1)[:, 0]
                loss_term_selec = tf.gather(loss_term, indices)
                Y_selec = tf.gather(Y_next, indices)

                local_loss = tf.zeros([nbSimul, self.ErgodicFactorModel.I], dtype=tf.float32)
                updates_loss = tf.square(Y_selec + loss_term_selec - self.ErgodicFactorModel.Y0)

                update_indices = tf.expand_dims(indices, axis=1)
                local_loss = tf.tensor_scatter_nd_update(local_loss, update_indices, updates_loss)

                indices_tau = tf.where(T_A == iStep+1)[:, 0]
                loss_term_tau = tf.gather(loss_term, indices_tau)

                updates_loss_tau = tf.square(loss_term_tau)

                update_indices_tau = tf.expand_dims(indices_tau, axis=1)
                local_loss = tf.tensor_scatter_nd_update(local_loss, update_indices_tau, updates_loss_tau)

                loss_tensor = loss_tensor.write(iStep+1, local_loss)

                Y_trans = Y_next

            loss_tensor = loss_tensor.stack()
            final_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(loss_tensor, axis=2), axis = 0))

            return final_loss


        @tf.function
        def trainOptNN(nbSimul, optimizerY, optimizerLambda):
            with tf.GradientTape(persistent=True) as tape:
                objFunc = optimizeBSDE(nbSimul)
            gradientsY = tape.gradient(objFunc, self.modelKerasY.trainable_variables)

            optimizerY.apply_gradients(zip(gradientsY, self.modelKerasY.trainable_variables))

            del tape
            return objFunc

        optimizerY = optimizers.Adam(learning_rate=self.lRate)
        optimizerLambda = optimizers.Adam(learning_rate=5*self.lRate)

        self.listlambd = []
        self.lossList = []
        self.duration = 0
        self.listlambd.append(self.modelKerasY.lambd.numpy())
        self.lossList.append(optimizeBSDE(batchSizeVal))
        for iout in range(num_epochExt):
            start_time = time.time()
            for epoch in range(num_epoch):
                # one stochastic gradient descent step
                trainOptNN(batchSize, optimizerY, optimizerLambda)
            end_time = time.time()
            rtime = end_time-start_time

            self.duration += rtime
            objError_Yterm = optimizeBSDE(batchSizeVal)
            lambd = self.modelKerasY.lambd.numpy()
            print(" Error",objError_Yterm.numpy(),  " elapsed time %5.3f s" % self.duration, "lambda so far ",lambd, 'epoch', iout)
            self.listlambd.append(lambd)
            self.lossList.append(objError_Yterm)


        return self.listlambd, self.lossList