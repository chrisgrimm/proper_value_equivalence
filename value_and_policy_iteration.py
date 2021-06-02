import numpy as np
from typing import Tuple, Union


def exact_policy_evaluation(
    gamma: float,
    policy: np.ndarray,
    rewards: np.ndarray,
    transitions: np.ndarray
) -> np.ndarray:
    num_states, num_actions = rewards.shape
    r_pi = np.sum(rewards * policy, axis=1)  # [s]
    p_pi = np.sum(policy[:, :, None] * transitions, axis=1)  # [s, s]
    v = np.linalg.inv((np.eye(num_states) - gamma*p_pi)) @ r_pi[:, None]
    return v[:, 0]


def run_value_iteration(
        gamma: float,
        rewards: np.ndarray,  # [s, a]
        transitions: np.ndarray,  # [s, a, s]
        initial_v: np.ndarray,  # [s]
        threshold: float = 1e-6,
        num_iters: int = np.inf,
        prob_update: float = 1.0,
        return_policy: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    num_states, num_actions = rewards.shape
    v = initial_v[:, None]  # [s, 1]
    p = np.transpose(transitions, [1, 0, 2])  # [a, s, s]
    r = np.transpose(rewards, [1, 0])[:, :, None]  # [a, s, 1]
    i = 0
    old_v = initial_v
    while i < num_iters:
        pv = (p @ v[None, :, :])  # [a, s, 1]
        va = r + gamma * pv
        pi = np.argmax(va, axis=0)[:, 0]  # [s]
        new_v = np.max(va, axis=0)  # [s, 1]
        err = np.max(np.abs(new_v[:, 0] - v[:, 0]), axis=0)
        if err <= threshold:
            break
        v = new_v
        i += 1
    additional_output = ()
    if prob_update < 1:
        update = (np.random.uniform(0, 1, size=[num_states]) < prob_update).astype(np.float32)
        new_v = new_v * update + old_v[:, None] * (1 - update)
        additional_output = (update,)
    if return_policy:
        new_onehot = np.zeros(shape=rewards.shape)
        for s in range(rewards.shape[0]):
            new_onehot[s, pi[s]] = 1
        return (new_v[:, 0], new_onehot) + additional_output
    else:
        return (new_v[:, 0],) + additional_output
