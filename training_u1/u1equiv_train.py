from collections import defaultdict
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("..")

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.python.keras.backend as K
import numpy as np
import sympy as sp
import time
import math
import argparse
from pathlib import Path

import MLGeometry as mlg
import models


z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]

parser = argparse.ArgumentParser()
# Data generation
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--n_pairs', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--function', default='f0')
parser.add_argument('--psi', type=float, default=0.5)
parser.add_argument('--phi', type=float)
parser.add_argument('--alpha', type=float)

# Network

parser.add_argument('--model_name', type=str, default='u1_model_relu')
parser.add_argument('--layers', type=str, default='10-10_1')
parser.add_argument('--g_steps', type=int, default=16)
parser.add_argument('--load_model')
parser.add_argument('--save_dir', default='experiments-marek')
parser.add_argument('--save_name', default='no_name')

# Training
parser.add_argument('--max_epochs', type=int, default=250)
parser.add_argument('--loss_func', default='weighted_MAPE')
parser.add_argument('--clip_threshold', type=float, default=None)
parser.add_argument('--optimizer', default='Adam')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--decay_rate', type=float, default=1.0)
parser.add_argument('--num_correction_pairs', type=int, default=10)

#DD@
parser.add_argument('--cache_folder', type=str, default='cache')
parser.add_argument('--mem_limit', type=int, default=16)


print_was = print
print = lambda *x, **y: print_was(*x, **(y | {'flush': True}))

args = parser.parse_args()


print("Processing model: " + args.save_name)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  print("Found gpus", gpus)
  if args.mem_limit is not None:
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=args.mem_limit*1024)])
        print(f"Memory limit set to {args.mem_limit}GB for each GPU")
    except RuntimeError as e:
        print("Memory limit not set ", e)
else:
    print("No gpus found")


# Data generation 
seed = args.seed
n_pairs = args.n_pairs
batch_size = args.batch_size
psi = args.psi

f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + psi*z0*z1*z2*z3*z4
if args.function == 'f1':
    phi = args.phi
    f = f + phi*(z3*z4**4 + z3**2*z4**3 + z3**3*z4**2 + z3**4*z4)
elif args.function == 'f2':
    alpha = args.alpha
    f = f + alpha*(z2*z0**4 + z0*z4*z1**3 + z0*z2*z3*z4**2 + z3**2*z1**3 + z4*z1**2*z2**2 + z0*z1*z2*z3**2 +
                   z2*z4*z3**3 + z0*z1**4 + z0*z4**2*z2**2 + z4**3*z1**2 + z0*z2*z3**3 + z3*z4*z0**3 + z1**3*z4**2 +
                   z0*z2*z4*z1**2 + z1**2*z3**3 + z1*z4**4 + z1*z2*z0**3 + z2**2*z4**3 + z4*z2**4 + z1*z3**4)

np.random.seed(seed)
tf.random.set_seed(seed)
cache_folder = Path(args.cache_folder) / f"cache_v3_s{seed}_n{n_pairs}"
cache_folder.mkdir(exist_ok=True)
cache_trainset_path = cache_folder / 'train_set'
cache_testset_path = cache_folder / 'test_set'

"""
We added cache for itermediate results
"""
if not all([
    cache_folder.exists(),
    cache_trainset_path.exists(),
    cache_testset_path.exists(),
    ]):

    HS = mlg.hypersurface.Hypersurface(Z, f, n_pairs)
    HS_test = mlg.hypersurface.Hypersurface(Z, f, n_pairs)

    train_set = mlg.tf_dataset.generate_dataset(HS)
    test_set = mlg.tf_dataset.generate_dataset(HS_test)

    #if batch_size is None or args.optimizer.lower() == 'lbfgs':
    if batch_size is None:
        batch_size = n_pairs

    train_set = train_set.shuffle(n_pairs)
    test_set = test_set.shuffle(n_pairs)

    train_set.save(str(cache_trainset_path))
    test_set.save(str(cache_testset_path))

train_set = tf.data.Dataset.load(str(cache_trainset_path))
test_set = tf.data.Dataset.load(str(cache_testset_path))

train_set = train_set.batch(batch_size)
test_set = test_set.batch(batch_size)

np.random.seed(seed+1)
tf.random.set_seed(seed+1)

#if batch_size is None or args.optimizer.lower() == 'lbfgs':
if batch_size is None:
    batch_size = n_pairs


# Network 

n_layers, m_layers = args.layers.split('-')

n_units = list(map(int, n_layers.split('_')))
m_units = list(map(int, m_layers.split('_')))


model_list = {
    "u1_model_relu": models.u1_model_relu,
    "u1_model_tanh": models.u1_model_tanh,
    "u1_model_simple": models.u1_model_simple,
}

load_path = args.load_model
if load_path is not None:
    model = tf.keras.models.load_model(load_path, compile=False)
else:
    model = model_list[args.model_name](n_units, m_units)


# model.build(inputs=tf.Tensor(shape=(None, 5), dtype=tf.complex64))



max_epochs = args.max_epochs
func_dict = {"weighted_MAPE": mlg.loss.weighted_MAPE, "weighted_MSE": mlg.loss.weighted_MSE, "max_error":mlg.loss.max_error,
             "MAPE_plus_max_error": mlg.loss.MAPE_plus_max_error}
loss_func = func_dict[args.loss_func]
#early_stopping = False
clip_threshold = args.clip_threshold
save_dir = Path(args.save_dir)
if not save_dir.exists():
    save_dir.mkdir(parents=True)
save_name = args.save_name


@tf.function
def volume_form(x, Omega_Omegabar, mass, restriction, fs_metric):
    kahler_metric = mlg.complex_math.complex_hessian(tf.math.real(model(x)), x)
    volume_form = tf.math.real(tf.linalg.det(fs_metric + tf.matmul(restriction, tf.matmul(kahler_metric, restriction, adjoint_b=True))))
    weights = mass / tf.reduce_sum(mass)
    factor = tf.reduce_sum(weights * volume_form / Omega_Omegabar)
    #factor = tf.constant(35.1774, dtype=tf.complex64)
    return volume_form / factor



def cal_total_loss(dataset, loss_function):

    total_loss = tf.constant(0, dtype=tf.float32)
    total_mass= tf.constant(0, dtype=tf.float32)

    for step, (points, Omega_Omegabar, mass, restriction, fs_metric) in enumerate(dataset):
        det_omega = volume_form(points, Omega_Omegabar, mass, restriction, fs_metric)
        mass_sum = tf.reduce_sum(mass)
        total_loss += loss_function(Omega_Omegabar, det_omega, mass) * mass_sum
        total_mass += mass_sum
    total_loss = total_loss / total_mass

    return total_loss.numpy()

def cal_max_error(dataset):
    '''
    find max|eta - 1| over the whole dataset: calculate the error on each batch then compare.
    '''
    max_error_tmp = 0
    for step, (points, Omega_Omegabar, mass, restriction, fs_metric) in enumerate(dataset):
        det_omega = volume_form(points, Omega_Omegabar, mass, restriction, fs_metric)
        error = mlg.loss.max_error(Omega_Omegabar, det_omega, mass).numpy()
        if error > max_error_tmp:
            max_error_tmp = error

    return max_error_tmp

# Training
start_time = time.time()
if args.optimizer.lower() == 'lbfgs':
    # iter+1 everytime f is evoked, which will also be invoked when calculationg the hessian, etc
    # So the true max_epochs will be 3 times user's input
    max_epochs = int(max_epochs/3)
    train_func = mlg.lbfgs.function_factory(model, loss_func, train_set)

    init_params = tf.dynamic_stitch(train_func.idx, model.trainable_variables)
    results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=train_func,
                                           initial_position=init_params,
                                           max_iterations=max_epochs,
                                           num_correction_pairs=args.num_correction_pairs)
    train_func.assign_new_model_parameters(results.position)

else:
    if args.optimizer.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(args.learning_rate)
    else:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps = n_pairs/batch_size,
            decay_rate = args.decay_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        #optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    train_log_dir = save_dir / 'logs' / save_name / 'train'
    test_log_dir = save_dir / 'logs' / save_name / 'test'
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
    test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))


    # class VolumeModel(tf.keras.Model):
    #     def __init__(self, model):
    #         super(VolumeModel, self).__init__()
    #         self.model = model

    #     @tf.function
    #     def call(self, inputs):
    #         x, Omega_Omegabar, mass, restriction, fs_metric = inputs
    #         y = self.model(x)
    #         kahler_metric = mlg.complex_math.complex_hessian(tf.math.real(y), x)
    #         volume_form = tf.math.real(tf.linalg.det(fs_metric + tf.matmul(restriction, tf.matmul(kahler_metric, restriction, adjoint_b=True))))
    #         weights = mass / tf.reduce_sum(mass)
    #         factor = tf.reduce_sum(weights * volume_form / Omega_Omegabar)
    #         #factor = tf.constant(35.1774, dtype=tf.complex64)
    #         return volume_form / factor

    # def map_func(x, Omega_Omegabar, mass, restriction, fs_metric):
    #     return (
    #         (x, Omega_Omegabar, mass, restriction, fs_metric),
    #         (Omega_Omegabar, mass)
    #     )    

    # train_set_t = train_set.map(map_func)
    # test_set_t = test_set.map(map_func)

    # volmodel = VolumeModel(model)
    # volmodel.compile(
    #     optimizer=optimizer, 
    #     loss=lambda y_true, y_pred: loss_func(y_true[0], y_pred, y_true[1])
    # )

    # x_sample, y_sample = list(train_set_t.take(1))[0]
    # y_tmp = volmodel(x_sample)
    # volmodel.summary()

    # volmodel.fit(
    #     train_set_t,
    #     epochs=max_epochs,
    #     validation_data=test_set_t,
    #     verbose=1
    # )

    # print("Done")
    # exit(0)

    stop = False
    loss_old = 100000
    epoch = 0



    while epoch < max_epochs and stop is False:
        epoch = epoch + 1
        train_loss = []
        mean_grads = defaultdict(list)
        for step, (points, Omega_Omegabar, mass, restriction, fs_metric) in enumerate(train_set):


            with tf.GradientTape() as tape:
                det_omega = volume_form(points, Omega_Omegabar, mass, restriction, fs_metric)
                loss = loss_func(Omega_Omegabar, det_omega, mass)
                grads = tape.gradient(loss, model.trainable_weights)

                for grad, var in zip(grads, model.trainable_weights):
                    if grad is None:
                        mean_grads[var.name + "-- None"].append(0)
                    else:
                        # give gradient norm
                        mean_grads[var.name].append(tf.norm(grad))

                if clip_threshold is not None:
                    grads = [tf.clip_by_value(grad, -clip_threshold, clip_threshold) for grad in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_loss.append(loss.numpy())
        
        if epoch == 1:
            for w in model.trainable_weights:
                print(w.name, w.shape)

        # evaluation after every round of training
        test_loss = cal_total_loss(test_set, loss_func)
        print("epoch:", epoch, "test_loss:", test_loss, "train_loss:", np.mean(train_loss))
        for key in mean_grads.keys():
            print(key, np.mean(mean_grads[key]))

        if epoch % 10 == 0:
            sigma_max_train = cal_max_error(train_set) 
            sigma_max_test = cal_max_error(test_set) 

            E_train = cal_total_loss(train_set, mlg.loss.weighted_MSE)
            E_test = cal_total_loss(test_set, mlg.loss.weighted_MSE)
        
            sigma_train = cal_total_loss(train_set, mlg.loss.weighted_MAPE)
            sigma_test  = cal_total_loss(test_set, mlg.loss.weighted_MAPE)

            def delta_sigma_square_train(y_true, y_pred, mass):
                weights = mass / K.sum(mass)
                return K.sum((K.abs(y_true - y_pred) / y_true - sigma_train)**2 * weights)

            def delta_sigma_square_test(y_true, y_pred, mass):
                weights = mass / K.sum(mass)
                return K.sum((K.abs(y_true - y_pred) / y_true - sigma_test)**2 * weights)

            delta_sigma_train = math.sqrt(cal_total_loss(train_set, delta_sigma_square_train) / n_pairs)
            delta_sigma_test = math.sqrt(cal_total_loss(test_set, delta_sigma_square_test) / n_pairs)

            # print("train_loss:", loss.numpy())
            # print("test_loss:", test_loss)

            with train_summary_writer.as_default():
                tf.summary.scalar('max_error', sigma_max_train, step=epoch)
                tf.summary.scalar('delta_sigma', delta_sigma_train, step=epoch)
                tf.summary.scalar('E', E_train, step=epoch)
                tf.summary.scalar('sigma', sigma_train , step=epoch)

            with test_summary_writer.as_default():
                tf.summary.scalar('max_error', sigma_max_test, step=epoch)
                tf.summary.scalar('delta_sigma', delta_sigma_test, step=epoch)
                tf.summary.scalar('E', E_test, step=epoch)
                tf.summary.scalar('sigma', sigma_test, step=epoch)    # Early stopping 

       # if early_stopping is True and epoch > 800:
       #     if epoch % 5 == 0:
       #         if train_loss > loss_old:
       #             stop = True 
       #         loss_old = train_loss 

train_time = time.time() - start_time

model.save(save_dir / save_name)

sigma_train = cal_total_loss(train_set, mlg.loss.weighted_MAPE) 
sigma_test = cal_total_loss(test_set, mlg.loss.weighted_MAPE) 
E_train = cal_total_loss(train_set, mlg.loss.weighted_MSE) 
E_test = cal_total_loss(test_set, mlg.loss.weighted_MSE) 
sigma_max_train = cal_max_error(train_set) 
sigma_max_test = cal_max_error(test_set) 

#######################################################################
# Calculate delta_sigma

def delta_sigma_square_train(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum((K.abs(y_true - y_pred) / y_true - sigma_train)**2 * weights)

def delta_sigma_square_test(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum((K.abs(y_true - y_pred) / y_true - sigma_test)**2 * weights)

def delta_E_square_train(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum(((y_pred / y_true - 1)**2 - E_train)**2 * weights)

def delta_E_square_test(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum(((y_pred / y_true - 1)**2 - E_test)**2 * weights)

delta_sigma_train = math.sqrt(cal_total_loss(train_set, delta_sigma_square_train) / n_pairs)
delta_sigma_test = math.sqrt(cal_total_loss(test_set, delta_sigma_square_test) / n_pairs)
delta_E_train = math.sqrt(cal_total_loss(train_set, delta_E_square_train) / n_pairs)
delta_E_test = math.sqrt(cal_total_loss(test_set, delta_E_square_test) / n_pairs)

#print(delta_sigma_train)
#print(delta_sigma_test)

#####################################################################
# Write to file

with open(save_dir / save_name / ".txt", "w") as f:
    f.write('[Results] \n')
    f.write('model_name = {} \n'.format(save_name))
    f.write('seed = {} \n'.format(seed))
    f.write('n_pairs = {} \n'.format(n_pairs))
    f.write('n_pairs = {} \n'.format(n_pairs))
    f.write('batch_size = {} \n'.format(batch_size))
    f.write('function = {} \n'.format(args.function))
    f.write('psi = {} \n'.format(psi))
    if args.function == 'f1':
        f.write('phi = {} \n'.format(phi))
    elif args.function == 'f2':
        f.write('alpha = {} \n'.format(alpha)) 
    f.write('k = {} \n'.format(k)) 
    f.write('n_parameters = {} \n'.format(model.count_params())) 
    f.write('loss function = {} \n'.format(loss_func.__name__))
    if clip_threshold is not None:
        f.write('clip_threshold = {} \n'.format(clip_threshold))
    f.write('\n')
    f.write('n_epochs = {} \n'.format(max_epochs))
    f.write('train_time = {:.6g} \n'.format(train_time))
    f.write('sigma_train = {:.6g} \n'.format(sigma_train))
    f.write('sigma_test = {:.6g} \n'.format(sigma_test))
    f.write('delta_sigma_train = {:.6g} \n'.format(delta_sigma_train))
    f.write('delta_sigma_test = {:.6g} \n'.format(delta_sigma_test))
    f.write('E_train = {:.6g} \n'.format(E_train))
    f.write('E_test = {:.6g} \n'.format(E_test))
    f.write('delta_E_train = {:.6g} \n'.format(delta_E_train))
    f.write('delta_E_test = {:.6g} \n'.format(delta_E_test))
    f.write('sigma_max_train = {:.6g} \n'.format(sigma_max_train))
    f.write('sigma_max_test = {:.6g} \n'.format(sigma_max_test))

with open(save_dir / "summary.txt", "a") as f:
    if args.function == 'f0':  
        f.write('{} {} {} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g}\n'.format(save_name, args.function, psi, train_time, sigma_train, sigma_test, E_train, E_test, sigma_max_train, sigma_max_test))
    elif args.function == 'f1':  
        f.write('{} {} {} {} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g}\n'.format(save_name, args.function, psi, phi, train_time, sigma_train, sigma_test, E_train, E_test, sigma_max_train, sigma_max_test))
    elif args.function == 'f2': 
        f.write('{} {} {} {} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g}\n'.format(save_name, args.function, psi, alpha, train_time, sigma_train, sigma_test, E_train, E_test, sigma_max_train, sigma_max_test))
