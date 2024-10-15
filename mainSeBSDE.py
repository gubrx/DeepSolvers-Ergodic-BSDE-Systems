import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ErgodicFactorModel import Example1, ErgodicPower, ErgodicExponentiel
from Networks import Net
from StochasticFactor import StochasticFactor, OrnsteinUhlenbeck
from SolverSeBSDE import SolverGlobaleBSDE, SolverLocaleBSDE, SolverADLocaleBSDE


# Parameters
T_H = 1
h = 0.01
kappa = [1.3]
dim = len(kappa)
I = 2
Q = np.array([[-0.01, 0.01], [0.1, -0.1]])
q = np.min(Q[Q>=0])
is_decoupled = (q == 0.0)
init_state = 1
nu = 1.
x0 = 1.

# Power utility examples parameters
delt = 0.25  # risk aversion
p = [[1.2], [0.7]]  # List of Lipschitz constant market price of risk function of states - each of dimension dim
b = 3  # born market price of risk
gamma = 0.2

def market_price(i, p, v):
    v = tf.convert_to_tensor(v, dtype=tf.float32)
    pa = tf.convert_to_tensor(p[i], dtype=tf.float32)
    v = tf.reshape(v, (-1, 1))
    pv = pa * v

    norm_pv = tf.norm(pv, axis=1, keepdims=True)
    condition = norm_pv > b
    pv = tf.where(condition, (pv / norm_pv) * b, pv)

    return pv

# NN Parameters
nbNeuron = 20 + dim
nbLayer = 2
num_epochExt = 20
num_epoch = 20
batchSize = 100
lRate = 0.0003
activation = tf.nn.tanh

# Select example
example_name = input("Enter the example to run (Example1, ErgodicPower, ErgodicExponentiel)")
if example_name in ['Example1']:
    Cv = input("Enter a list of size I for 'Cv' (comma-separated values): ")
    Cv = [float(x) for x in Cv.split(',')]
elif example_name in ['ErgodicPower']:
    Cv = (delt/(1-delt))*np.max(p)*np.max([1, b])
    print('Cv =', Cv)
elif example_name in ['ErgodicExponentiel']:
    Cv = 2*np.max([1, b])*np.max(p)
    print('Cv =', Cv)

# Prompt for a valid value of muval
muval = None
while muval is None or muval <= np.max(Cv):
    try:
        muval_input = input(f"Enter a value for 'muval' greater than {np.max(Cv)}: ")
        muval = float(muval_input)
        if muval <= np.max(Cv):
            print(f"The value of 'muval' must be greater than {np.max(Cv)}. You entered: {muval}")
    except ValueError:
        print("Please enter a valid numeric value.")

print(f"'muval' set to: {muval}")

solver_name = input("Method (Global, Local, ADLocal): ")

stochastic_factor = OrnsteinUhlenbeck(x0, h, T_H, dim, muval, nu, kappa)
if example_name == 'Example1':
    example = Example1(stochastic_factor, Cv, I, init_state, Q)
elif example_name == 'ErgodicPower':
    example = ErgodicPower(stochastic_factor, market_price, p, b, delt, I, init_state, Q)
elif example_name == 'ErgodicExponentiel':
    example = ErgodicPower(stochastic_factor, market_price, p, b, gamma, I, init_state, Q)

# Some values
lambda_lim = example.lambdlim
if hasattr(example, 'lambd_ex'):
    print(f'lambda exact= {example.lambd_ex}')

# Neural network
layerSize = nbNeuron * np.ones((nbLayer,), dtype=np.int32)
if solver_name == 'Global':
  kerasModel = Net(True, lambda_lim, dim, I, layerSize, activation, is_decoupled)
  solver = SolverGlobaleBSDE(example, kerasModel, lRate)
  print(f"Trainable variables before initialization: {kerasModel.trainable_variables}")

if solver_name == 'Local':
  kerasModelY = Net(False, lambda_lim, 1, I, layerSize, activation, is_decoupled)
  kerasModelZ = Net(True, lambda_lim, dim, I, layerSize, activation, is_decoupled)
  solver = SolverLocaleBSDE(example, kerasModelY, kerasModelZ, lRate)
  print(f"Trainable variables before initialization: {kerasModelY.trainable_variables} {kerasModelZ.trainable_variables}")

if solver_name == 'ADLocal':
  kerasModelY = Net(True, lambda_lim, 1, I, layerSize, activation, is_decoupled)
  solver = SolverADLocaleBSDE(example, kerasModelY, lRate)
  print(f"Trainable variables before initialization: {kerasModelY.trainable_variables}")


# train and  get solution
lambdlist, lossT_Hlist = solver.train(batchSize, batchSize*100, num_epoch, num_epochExt)

if solver_name == 'Global':
  print(f"Trainable variables after initialization: {kerasModel.trainable_variables}")
  print(f"Model created: {kerasModel}")
  print(f"Model type: {type(kerasModel)}")
  kerasModel.summary()
if solver_name == 'Local':
  print(f"Model created: {kerasModelY} {kerasModelZ}")
  print(f"Model type: {type(kerasModelY)} {type(kerasModelZ)}")
  print(f"Trainable variables after initialization: {kerasModelY.trainable_variables} {kerasModelZ.trainable_variables}")
  kerasModelY.summary()
  kerasModelZ.summary()

###### PLOTS ######
if solver_name == 'Global':
  loss_name = r'$L^{B_{\epsilon}}(\theta, \bar{\lambda})$'
  solver_label = 'GeBSDE'

if solver_name == 'Local':
  loss_name = r'$L_{loc}^{B_{\epsilon}}(\theta, \bar{\lambda})$'
  solver_label = 'LAeBSDE'

Nepoch = range(0, num_epoch*(num_epochExt+1), num_epoch)
plt.figure()
plt.plot(Nepoch, lossT_Hlist, label=f"{loss_name} - {solver_label}")
plt.grid(True, which = 'both', linestyle='--', alpha=0.7)
plt.legend()
plt.xlabel('Number of Epochs')
plt.tight_layout()
plt.yscale('log')
plt.title('Loss training result')
plt.show()

plt.figure()
plt.plot(Nepoch, lambdlist, label=rf'$\bar{{\lambda}}$ - {solver_label}')
plt.grid(True, which = 'both', linestyle='--', alpha=0.7)
plt.legend()
plt.xlabel('Number of Epochs')
plt.tight_layout()
plt.title('Convergence lambda training result')
plt.show()