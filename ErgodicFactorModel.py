import tensorflow as tf
import numpy as np
import scipy.optimize as opt

class ErgodicFactorModel():
  def __init__(self, stochastic_factor, I, init_state, Q):
    self.stochastic_factor = stochastic_factor
    self.dt = stochastic_factor.dt
    self.T_H = stochastic_factor.T_H
    self.dim = stochastic_factor.dim
    self.I = I
    self.Q = Q
    self.init_state = init_state

  def proj_Pi(self, Pi, X):
      Pi = tf.convert_to_tensor(Pi, dtype=X.dtype)
      # Ensure Pi has shape (dim, 2)
      assert Pi.shape[1] == 2, "Pi should have shape (dim, 2)"

      Pi_expanded = tf.expand_dims(Pi, 0)

      X_expanded = tf.expand_dims(X, -1)
      lower_mask = X_expanded < Pi_expanded[..., 0:1]
      upper_mask = X_expanded > Pi_expanded[..., 1:2]

      X_proj = tf.where(lower_mask, Pi_expanded[..., 0:1], X_expanded)
      X_proj = tf.where(upper_mask, Pi_expanded[..., 1:2], X_proj)

      X_proj = tf.squeeze(X_proj, -1)

      return X_proj

  def squared_dist(self, Pi, x):
    # Ensure Pi is of shape (dim, 2)
    Pi = tf.convert_to_tensor(Pi, dtype=tf.float32)

    x = tf.cast(x, dtype=tf.float32)
    if Pi.ndim == 1:
        Pi = Pi[tf.newaxis, :]
    if x.ndim == 1:
        x = tf.expand_dims(x, axis=1)

    lower_bounds = tf.expand_dims(Pi[:, 0], axis=0)  # shape (1, dim)
    upper_bounds = tf.expand_dims(Pi[:, 1], axis=0) 

    lower_bounds = tf.tile(lower_bounds, [tf.shape(x)[0], tf.shape(x)[1]])  # shape (nbSimul, dim)
    upper_bounds = tf.tile(upper_bounds, [tf.shape(x)[0], tf.shape(x)[1]]) 

    below_lower = tf.maximum(lower_bounds - x, 0.0)
    above_upper = tf.maximum(x - upper_bounds, 0.0)

    total_distance = tf.reduce_sum(below_lower**2 + above_upper**2, axis=1)

    return total_distance

  def next_jump_Poiss(self, i, Q):
      L = [np.random.exponential(1/Q[i-1, j]) if j != i-1 else np.inf for j in range(self.I)]
      delta_Ti = min(L)

      return delta_Ti, L.index(delta_Ti)

  def switching_seq(self, i, Q, Tmax):
      T = 0.0
      Tlist = [0]
      S = [i]
      while T < Tmax:
        delta, j = self.next_jump_Poiss(S[-1], Q)
        T += delta
        Tlist.append(T)
        S.append(j+1)

      return Tlist, S


  @tf.function
  def forward_BSDE(self, T_A, dW, V, Q, ergodic_model, kerasModel):
      nbr_traj = len(V[0,:])
      maxind = tf.reduce_max(T_A)

      Y_sim = tf.TensorArray(dtype=tf.float32, size=int(maxind+1))
      Z_sim = tf.TensorArray(dtype=tf.float32, size=int(maxind))

      Y_trans = ergodic_model.Y0 * tf.ones([nbr_traj, ergodic_model.I], dtype=tf.float32)
      Y_sim = Y_sim.write(0, Y_trans)

      epsilon = 1e-8

      for iStep in tf.range(int(maxind)):
          step_start_time = tf.timestamp()

          input_tensor = tf.expand_dims(V[iStep, :], axis=-1)
          Z = kerasModel(True, input_tensor)

          interaction_terms = [
              tf.reduce_sum(
              ergodic_model.Q[i, :] * (tf.exp(tf.clip_by_value(Y_trans - Y_trans[:, i:i+1], - ergodic_model.CY, ergodic_model.CY)) - 1.0), axis=1) for i in range(ergodic_model.I)]
          interaction_terms = tf.stack(interaction_terms, axis=1)

          Y_next = Y_trans - self.dt * ergodic_model.f_vectorized(ergodic_model.f, V[iStep, :], Z) + kerasModel.lambd*self.dt \
          - interaction_terms * self.dt + tf.reduce_sum(Z * dW[iStep, :, tf.newaxis], axis=1)

          # Stack updates and assign to Y_next
          Y_sim = Y_sim.write(iStep + 1, Y_next)
          Z_sim = Z_sim.write(iStep, Z)

          Y_trans = Y_next

      Y_sim = Y_sim.stack()
      Z_sim = Z_sim.stack()

      return Y_sim, Z_sim

  #construct the solution to the switching problem
  def switching_construction(self, T, Y_sim, Z_sim, jumptimes, states):
      maxind = len(Y_sim[:, 0])
      nbr_traj = len(Y_sim[0, :])
      T_list = []
      Y_switch = []
      Z_switch = []

      jumpindex = [int(jumptimes[j] / self.dt) for j in range(len(jumptimes))]
      print('jumpindex=', jumpindex)

      Y_post_jump = None

      for j in range(len(states) - 1):
          if jumpindex[j + 1] < maxind:
              # Calculate Y_pre_jump
              if jumpindex[j + 1] + 1 < maxind:
                  Y_pre_jump = Y_sim[jumpindex[j + 1], :, states[j] - 1] \
                              + ((jumptimes[j + 1] - jumpindex[j + 1] * self.dt) / self.dt) * \
                              (Y_sim[jumpindex[j + 1], :, states[j] - 1] - Y_sim[jumpindex[j + 1] + 1, :, states[j] - 1])
              else:
                  Y_pre_jump = Y_sim[jumpindex[j + 1], :, states[j] - 1]

              if j > 0:
                  Y_switch_add = Y_sim[jumpindex[j] + 1:jumpindex[j + 1] + 1, :, states[j] - 1]
                  Y_switch_add = np.insert(Y_switch_add, 0, Y_post_jump, axis=0)
                  Y_switch_add = np.insert(Y_switch_add, -1, Y_pre_jump, axis=0)
                  T_add = T[jumpindex[j] + 1:jumpindex[j + 1] + 1].tolist()
                  T_add.insert(0, jumptimes[j])
                  T_add.append(jumptimes[j + 1])
              else:
                  Y_switch_add = Y_sim[jumpindex[j]:jumpindex[j + 1] + 1, :, states[j] - 1]
                  T_add = T[jumpindex[j]:jumpindex[j + 1] + 1].tolist()
                  Y_switch_add = np.insert(Y_switch_add, -1, Y_pre_jump, axis=0)
                  T_add.append(jumptimes[j + 1])

              # Calculate Y_post_jump for the next iteration
              if jumpindex[j + 1] + 1 < maxind:
                  Y_post_jump = Y_sim[jumpindex[j + 1], :, states[j + 1] - 1] \
                                + ((jumptimes[j + 1] - jumpindex[j + 1] * self.dt) / self.dt) * \
                                (Y_sim[jumpindex[j + 1], :, states[j + 1] - 1] - Y_sim[jumpindex[j + 1] + 1, :, states[j + 1] - 1])
              else:
                  Y_post_jump = Y_sim[jumpindex[j + 1], :, states[j + 1] - 1]

              Y_switch.append(Y_switch_add)
              T_list.append(T_add)

              # Construct Z_switch
              Z_switch_add = Z_sim[jumpindex[j]:jumpindex[j + 1] + 1, :, :, states[j] - 1]
              Z_switch.append(Z_switch_add)

      return T_list, Y_switch, Z_switch

  def f_vectorized(self, f, v, z):
      result = tf.TensorArray(tf.float32, size=self.I)

      for i in range(self.I):
          z_i = tf.gather(z, i, axis =2)
          result = result.write(i, f(i, v, z_i))
      result = result.stack()  # Shape [I, nbSimul]

      return tf.transpose(result)  # Shape [nbSimul, I]

class EV(ErgodicFactorModel):
  def __init__(self, stochastic_factor, I, init_state, Q):
      super().__init__(stochastic_factor, I, init_state, Q)
      if I==1:
        self.qmin = 1
      elif np.all(Q == 0):
        self.qmin = 1
      else:
        self.qmin = np.min(Q[Q > 0])
      self.dt = stochastic_factor.dt
      self.muval = stochastic_factor.muval
      self.kappa = stochastic_factor.kappa

class Example1(EV):
  def __init__(self, ornstein_uhlenbeck, Cv, I, init_state, Q):
      super().__init__(ornstein_uhlenbeck, I, init_state, Q)
      self.Cv = Cv
      self.Cvmax = np.max(self.Cv)
      self.Zlim = tf.norm(tf.constant(self.kappa, dtype=tf.float32))*(self.Cvmax / (self.muval - self.Cvmax))
      self.lambdlim = self.Cvmax*tf.exp(-1/2)
      self.CY = self.lambdlim / self.qmin
      print('CY=', self.CY)
      self.Y0 = [0.0, 0.1]
      if I ==1:
        self.Y0 = 0.2

  # Driver
  def f(self, i, v, z):
      return v*self.Cv[i]*tf.exp(-v**2/2)


class ErgodicPower(EV):
  def __init__(self, stochastic_factor, market_price, p, b, delt, I, init_state, Q):
      super().__init__(stochastic_factor, I, init_state, Q)
      self.delt = delt #risk aversion
      self.gamma = 1 /(1 - self.delt)
      self.market_price = market_price
      self.lambd_ex = 'Not known'
      self.p = p
      self.b = b

      self.Y0 = [1., 1.]
      self.lambdlim = (delt/(1-delt))*b**2
      self.Cv = (delt/(1-delt))*np.max(p)*np.max([1, b]) # (Facteur 2 si \Pi different de R^{d})
      print('Cv=', self.Cv)
      self.Cz = (delt/(1-delt))*np.max([1, 2*b]) + (1/2)
      self.CY = (1 / self.qmin)*(self.lambdlim + (self.Cv*self.muval*self.Cz)/(self.muval - self.Cv)**2)
      print('CY=', self.CY)
      print('norm kappa:', tf.norm(tf.constant(self.kappa, dtype=tf.float32)))
      self.Zlim = tf.norm(tf.constant(self.kappa, dtype=tf.float32))*(self.Cv / (self.muval - self.Cv))
      print('Zlim=', self.Zlim)
      self.Pi = np.array([[[-np.inf, np.inf]], [[-np.inf, np.inf]]])

  def f(self, i, v, z):
      return (1/2)*self.delt/(1 - self.delt)*tf.norm(z + self.market_price(i, self.p, v), axis = -1)**2 + (1/2)*tf.norm(z, axis=-1)**2

  # optimal portoflio
  def optimal_strategy(self, i, Pi, V, Z_list):
      thetV = self.market_price(i, self.p, V)

      return self.proj_Pi(Pi[i], (1/(1-self.delt))*(thetV + Z_list))

"""
  # Driver
  def f(self, i, v, z):
    return (self.delt*(self.delt - 1)/2)*self.squared_dist(self.Pi[i, :], (z + self.market_price(i, self.p, v))/(1 - self.delt)) + \
     (1/2)*self.delt/(1 - self.delt)*tf.norm(z + self.market_price(i, self.p, v), axis = -1)**2 + (1/2)*tf.norm(z, axis=-1)**2
"""

class ErgodicExponentiel(ErgodicFactorModel):
  def __init__(self, stochastic_factor, market_price, p, b, gamma, I, init_state, Q):
      super().__init__(stochastic_factor, I, init_state, Q)
      self.gamma = gamma
      self.market_price = market_price
      self.lambd_ex = 'Not known'
      self.p = p
      self.b = b

      self.Y0 = [1., 1.1]
      self.lambdlim = self.b**2
      self.Cv = 2*np.max([1, self.b])*np.max(p)
      self.Cz = np.max([3/2, 2*self.b])
      self.CY = (1 / self.qmin)*(self.lambdlim + (self.Cv*self.muval*self.Cz)/(self.muval - self.Cv)**2)
      self.Zlim = tf.norm(tf.constant(self.kappa, dtype=tf.float32))*(self.Cv / (self.stochastic_factor.muval - self.Cv))
      self.Pi = np.array([[[-np.inf, np.inf]], [[-np.inf, np.inf]]])

  # Driver
  def f(self, i, v, z):
    return (self.gamma**2/2)*self.squared_dist(self.Pi, (z + self.market_price(i, self.p, v))/self.gamma) - \
     (1/2)*tf.norm(z + self.market_price(i, self.p, v), axis = -1)**2 + (1/2)*tf.norm(z, axis=-1)**2
