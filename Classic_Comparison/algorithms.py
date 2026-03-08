# algorithms.py
import numpy as np
from config import *
from environment import get_state, step

def get_Q(Q, state):
    if state not in Q:
        Q[state] = np.zeros(N_ACTIONS)
    return Q[state]

def select_action(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTIONS)
    return np.argmax(get_Q(Q, state))

# --- Q-learning ---
def run_episode_qlearning(maze, Q, epsilon, alpha, gamma):
    pos, last_action = START, None
    state            = get_state(maze, pos, last_action)
    total_reward     = 0
    visit_count      = {}

    for _ in range(MAX_STEPS):
        action              = select_action(Q, state, epsilon)
        new_pos, reward, done = step(maze, pos, action)
        new_state           = get_state(maze, new_pos, action)

        count   = visit_count.get(new_pos, 0)
        reward += R_REVISIT * count
        visit_count[new_pos] = count + 1

        current_Q = get_Q(Q, state)[action]
        target_Q  = reward if done else reward + gamma * np.max(get_Q(Q, new_state))
        get_Q(Q, state)[action] += alpha * (target_Q - current_Q)

        pos, state, last_action = new_pos, new_state, action
        total_reward += reward
        if done:
            break

    return Q, total_reward

# --- SARSA ---
def run_episode_sarsa(maze, Q, epsilon, alpha, gamma):
    pos, last_action = START, None
    state            = get_state(maze, pos, last_action)
    action           = select_action(Q, state, epsilon)
    total_reward     = 0
    visit_count      = {}

    for _ in range(MAX_STEPS):
        new_pos, reward, done = step(maze, pos, action)
        new_state             = get_state(maze, new_pos, action)
        next_action           = select_action(Q, new_state, epsilon)

        count   = visit_count.get(new_pos, 0)
        reward += R_REVISIT * count
        visit_count[new_pos] = count + 1

        current_Q = get_Q(Q, state)[action]
        target_Q  = reward if done else reward + gamma * get_Q(Q, new_state)[next_action]
        get_Q(Q, state)[action] += alpha * (target_Q - current_Q)

        pos, state, action, last_action = new_pos, new_state, next_action, action
        total_reward += reward
        if done:
            break

    return Q, total_reward

# --- TD(lambda) ---
def run_episode_td_lambda(maze, Q, epsilon, alpha, gamma, lam=0.9):
    pos, last_action = START, None
    state            = get_state(maze, pos, last_action)
    total_reward     = 0
    visit_count      = {}

    # eligibility traces — same structure as Q
    E = {}
    def get_E(state):
        if state not in E:
            E[state] = np.zeros(N_ACTIONS)
        return E[state]

    for _ in range(MAX_STEPS):
        action              = select_action(Q, state, epsilon)
        new_pos, reward, done = step(maze, pos, action)
        new_state           = get_state(maze, new_pos, action)

        count   = visit_count.get(new_pos, 0)
        reward += R_REVISIT * count
        visit_count[new_pos] = count + 1

        td_error = reward - get_Q(Q, state)[action]
        if not done:
            td_error += gamma * np.max(get_Q(Q, new_state))

        # update eligibility trace for current state-action
        get_E(state)[action] += 1.0

        # update ALL states in trace
        for s in E:
            Q_s = get_Q(Q, s)
            Q_s += alpha * td_error * E[s]
            E[s] *= gamma * lam   # decay trace

        pos, state, last_action = new_pos, new_state, action
        total_reward += reward
        if done:
            break

    return Q, total_reward

# --- Monte Carlo (TD lambda=1 equivalent) ---
def run_episode_montecarlo(maze, Q, epsilon, alpha, gamma):
    pos, last_action = START, None
    state            = get_state(maze, pos, last_action)
    trajectory       = []   # store (state, action, reward)
    visit_count      = {}

    for _ in range(MAX_STEPS):
        action              = select_action(Q, state, epsilon)
        new_pos, reward, done = step(maze, pos, action)
        new_state           = get_state(maze, new_pos, action)

        count   = visit_count.get(new_pos, 0)
        reward += R_REVISIT * count
        visit_count[new_pos] = count + 1

        trajectory.append((state, action, reward))
        pos, state, last_action = new_pos, new_state, action

        if done:
            break

    # backward update from end of episode
    G = 0.0
    for state_t, action_t, reward_t in reversed(trajectory):
        G = reward_t + gamma * G
        current_Q = get_Q(Q, state_t)[action_t]
        get_Q(Q, state_t)[action_t] += alpha * (G - current_Q)

    return Q, sum(r for _, _, r in trajectory)