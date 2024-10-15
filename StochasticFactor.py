import tensorflow as tf

class StochasticFactor():
    def __init__(self, x0, dt, T_H, dim):
        self.x0 = x0
        self.dt = dt
        self.T_H = T_H
        self.dim = dim

    def one_step(self, V, mu, kappa):
        dw_sample = tf.sqrt(self.dt) * tf.random.normal(shape=(tf.shape(V)[0], self.dim))
        V = V + mu(V) * self.dt + tf.reduce_sum(tf.convert_to_tensor(kappa) * dw_sample, axis=1)
        return dw_sample, V

    @tf.function
    def sample(self, mu, kappa, num_sample):
        print("Tracing sample...")
        V = tf.ones((1, num_sample)) * self.x0
        dW = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        T_A = tf.zeros([num_sample], dtype=tf.float32)
        indT_H = tf.cast(tf.math.ceil(self.T_H / self.dt), tf.int32)

        shape_invariant_V = tf.TensorShape([None, num_sample])
        shape_invariant_dW = tf.TensorShape(None)

        for i in tf.range(indT_H):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(V, shape_invariant_V), ])
            dw_sample, V_new = self.one_step(V[-1, :], mu, kappa)
            V = tf.concat([V, V_new[tf.newaxis, :]], axis=0)
            dW = dW.write(i, dw_sample)

        Lstart = V[-1, :] - self.x0 * tf.ones(num_sample)
        i = indT_H +1

        def second_loop_body(i, V, dW, T_A, Lstart):
            V_last = V[-1, :]
            dw_sample, V_new = self.one_step(V_last, mu, kappa)

            V = tf.concat([V, V_new[tf.newaxis, :]], axis=0)
            dW = dW.write(i, dw_sample)

            L = V_new - self.x0
            condition = L * Lstart <= 0

            T_A = tf.where((condition) & (T_A == 0.0), tf.cast(i, tf.float32), T_A)

            return i + 1, V, dW, T_A, Lstart

        i, V, dW, T_A, _ = tf.while_loop(
            lambda i, V, dW, T_A, Lstart: tf.reduce_any(T_A == 0.0) & (i < tf.cast(20 / self.dt, tf.int32)),
            second_loop_body,
            loop_vars=[i, V, dW, T_A, Lstart],
            shape_invariants=[tf.TensorShape([]), shape_invariant_V, shape_invariant_dW, T_A.shape, Lstart.shape]
        )

        if i == 20 * indT_H:
            tf.print("The maximum return time over num_sample is large, greater than T=20. Consider changing parameters of the stochastic factor.")

        return tf.cast(T_A, tf.int32), dW.stack(), V


    def simple_sample(self, mu, kappa, num_sample):
        V = tf.ones((1, num_sample)) * self.x0
        indT_H = tf.cast(tf.math.ceil(self.T_H / self.dt), tf.int32)

        dW = tf.TensorArray(dtype=tf.float32, size=10*indT_H, clear_after_read=False)

        for i in tf.range(10 * indT_H):
            dw_sample, V_new = self.one_step(V[-1, :], mu, kappa)
            V = tf.concat([V, V_new[tf.newaxis, :]], axis=0)
            dW = dW.write(i, dw_sample)

        return dW.stack(), V


class OrnsteinUhlenbeck(StochasticFactor):
    def __init__(self, x0, dt, T_H, dim, muval, nu, kappa):
        super().__init__(x0, dt, T_H, dim)
        self.muval = muval
        self.kappa = kappa
        self.x0 = x0
        self.nu = nu

    def mu(self, x):
      return -self.muval * (x - self.nu)