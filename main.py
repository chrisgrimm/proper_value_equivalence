import jax
import jax.numpy as jnp
import jax.random as jrng
import optax
from ray import tune
import random
import numpy as np
import pickle
import os
import sys
import gridworld
import policies as policies_module
import value_and_policy_iteration as vpi


def init_model(key, num_states, num_actions, config):
    key, r_key, d_key, k_key = jrng.split(key, 4)
    uni_init = config['uni_init']
    r = jrng.uniform(r_key, [num_states, num_actions], minval=-1, maxval=1)
    if config['restrict_capacity']:
        d = jrng.uniform(d_key, [num_actions, num_states, config['model_rank']], minval=-uni_init, maxval=uni_init)
        k = jrng.uniform(k_key, [num_actions, config['model_rank'], num_states], minval=-uni_init, maxval=uni_init)
        return key, (r, d, k)
    else:
        p = jrng.uniform(d_key, [num_actions, num_states, num_states], minval=-uni_init, maxval=uni_init)
        return key, (r, p)


def params_to_model(params, config):
    if config['restrict_capacity']:
        r, d, k = params
        pd = jax.nn.softmax(d, axis=2)
        pk = jax.nn.softmax(k, axis=2)
        p = pd @ pk
    else:
        r, p = params
        p = jax.nn.softmax(p, axis=2)
    return r, jnp.transpose(p, [1, 0, 2])


def bellman_update(pi, v, r, p, gamma):
    # pi : [num_states, num_actions]
    # v : [num_states]
    # r : [num_states, num_actions]
    # p : [num_states, num_actions, num_states]
    vv = jnp.transpose(p, [1, 0, 2]) @ v[None, :, None]  # [num_actions, num_states, 1]
    rvv = r + gamma * jnp.transpose(vv[:, :, 0], [1, 0])  # [num_states, num_actions]
    return jnp.sum(pi * rvv, axis=1)  # [num_states]


def n_step_bellman_update(pi, v, r, p, n, gamma):
    for i in range(n):
        v = bellman_update(pi, v, r, p, gamma)
    return v


def ve_loss(params, pi_batch, v_batch, true_r, true_p, config):
    r, p = params_to_model(params, config)
    T = jax.vmap(n_step_bellman_update, (0, 0, None, None, None, None), 0)
    tv_model = T(pi_batch, v_batch, r, p, config['ve_mode'][0], config['gamma'])
    tv = T(pi_batch, v_batch, true_r, true_p, config['ve_mode'][0], config['gamma'])
    return jnp.mean(jnp.sum(jnp.square(tv - tv_model), axis=1), axis=0)


def fpve_loss(params, pi_batch, true_v_pi_batch, config):
    r, p = params_to_model(params, config)
    T = jax.vmap(bellman_update, (0, 0, None, None, None), 0)
    tv_model = T(pi_batch, true_v_pi_batch, r, p, config['gamma'])
    return jnp.mean(jnp.sum(jnp.square(tv_model - true_v_pi_batch), axis=1), axis=0)


def update_ve(params, state, opt, pi, v, true_r, true_p, config):
    loss, grads = jax.value_and_grad(ve_loss)(params, pi, v, true_r, true_p, config)
    updates, state = opt.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss


def update_fpve(params, state, opt, pi, true_v_pi, config):
    loss, grads = jax.value_and_grad(fpve_loss)(params, pi, true_v_pi, config)
    updates, state = opt.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss


def run_experiment(config):
    # set up seed
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    # construct env and get env params
    env = gridworld.FourRooms(p_intended=0.8)
    true_r = env.get_reward_matrix()
    true_p = env.get_transition_tensor()
    num_states, num_actions = np.shape(true_r)
    # initialize jax stuff
    key = jrng.PRNGKey(config['seed'])
    key, model_params = init_model(key, num_states, num_actions, config)
    opt = optax.adam(config['learning_rate'])
    state = opt.init(model_params)

    num_ve_steps, ve_policy_mode = config['ve_mode'] 

    def _update_ve(params, state, pi, v):
        return update_ve(params, state, opt, pi, v, true_r, true_p, config)
    _update_ve = jax.jit(_update_ve)

    def _update_fpve(params, state, pi, true_v_pi):
        return update_fpve(params, state, opt, pi, true_v_pi, config)
    _update_fpve = jax.jit(_update_fpve)

    # collect policies and values 
    if not config['use_vip']:
        policies, values = policies_module.collect_random_policies(
            100_000, ve_policy_mode, num_ve_steps, num_states, num_actions, true_r, true_p, config['gamma'])
    else:
        policies, values = policies_module.collect_iteration_policies(
            1000, 100, ve_policy_mode, num_ve_steps, num_states, num_actions, true_r, true_p, config['gamma'])

    stored_models = []
    stored_models_path = os.path.join(tune.get_trial_dir(), 'models.pickle')
    for ts in range(1, config['num_iters']+1):
        idx = np.random.randint(0, len(policies), size=[config['batch_size']])
        pi_batch = policies[idx, :, :]
        v_batch = values[idx, :]
        if num_ve_steps == np.inf:
            model_params, state, loss = _update_fpve(model_params, state, pi_batch, v_batch)
        else:
            model_params, state, loss = _update_ve(model_params, state, pi_batch, v_batch)
        to_report = {}
        if ts % config['store_model_every'] == 0:
            r, p = params_to_model(model_params, config)
            stored_models.append((ts, r, p))
        if ts % config['store_loss_every'] == 0:
            to_report['loss'] = loss
        if ts % config['eval_model_every'] == 0:
            r, p = params_to_model(model_params, config)
            r, p = np.array(r), np.array(p)
            _, pi = vpi.run_value_iteration(
                config['gamma'], r, p, np.zeros([num_states]), threshold=1e-4, return_policy=True)
            value_pi = vpi.exact_policy_evaluation(config['gamma'], pi, true_r, true_p)
            to_report['mean_value'] = np.mean(value_pi)
            to_report['ts'] = ts
        
        if len(stored_models) > 0 and ts == (config['num_iters'] - 1):
            with open(stored_models_path, 'wb') as f:
                pickle.dump(stored_models, f) 
            to_report['model_path'] = stored_models_path
        if len(to_report) > 0:
            tune.report(**to_report)


if __name__ == '__main__':

    model_capacity_search_space = {
        'seed': tune.randint(0, 500_000),
        'gamma': 0.99,
        'batch_size': 50,
        've_mode': tune.grid_search([(np.inf, 'stoch'), (np.inf, 'det')]),
        'model_rank': tune.grid_search([20, 30, 40, 50, 60, 70, 80, 90, 100, 104]),
        'learning_rate': 5e-4,
        'eval_model_every': 10_000,
        'store_loss_every': 10_000,
        'num_iters': 1_000_000,
        'restrict_capacity': True,
        'store_model_every': np.inf,
        'uni_init': 5,
        'use_vip': True,
    }
   
    n_step_search = [1, 30, 40, 50, 60, np.inf]
    diameter_search_space = {
        'seed': tune.randint(0, 500_000),
        'gamma': 0.99,
        'batch_size': 50,
        've_mode': tune.grid_search([(i, 'det_and_stoch') for i in n_step_search]),
        'model_rank': 104,
        'learning_rate': 1e-3,
        'eval_model_every': np.inf,
        'store_loss_every': 100,
        'num_iters': 500_000,
        'restrict_capacity': False,
        'store_model_every': 1_000,
        'uni_init': 5,
        'use_vip': False,
    }

    mode = sys.argv[1]
    local_dir = sys.argv[2]
    seed = int(sys.argv[3])

    random.seed(seed)
    np.random.seed(seed)
    assert mode in ['diameter', 'capacity']
    space_samples_pairs = {'diameter': ([diameter_search_space], 120),
                           'capacity': ([model_capacity_search_space], 10)}
    spaces, num_samples = space_samples_pairs[mode]
    exp_dirs = []
    for space in spaces:
        analysis = tune.run(run_experiment,
                            num_samples=num_samples,
                            config=space,
                            local_dir=os.path.join(local_dir, f'ray_{mode}'),
                            resources_per_trial={'cpu': 0.5, 'gpu': 0.2},
                            fail_fast=True)
        exp_dirs.append(analysis._experiment_dir)
    print('Sweep directories:')
    for exp_dir in exp_dirs:
        print(exp_dir)


