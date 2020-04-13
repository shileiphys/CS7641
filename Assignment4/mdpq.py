import numpy as np
import gym
import itertools
import sys

# based on https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb
def value_iteration(env, gamma=1.0, theta=0.0001):
    """
    Sutton & Barto http://incompleteideas.net/book/the-book.html

    env: openAI env
    """
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                next_state_value = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    next_state_value += prob * (reward + gamma * V[next_state])
                action_values[a] = next_state_value
            best_action_value = np.max(action_values)
            V[s] = best_action_value

            delta = max(delta, np.abs(v - V[s]))
        if (delta < theta):
            print("converged")
            break

    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            next_state_value = 0
            for prob, next_state, reward, done in env.P[s][a]:
                next_state_value += prob * (reward + gamma * V[next_state])
            action_values[a] = next_state_value
        best_action = np.argmax(action_values)
        policy[s, best_action] = 1.0

    return policy, V


def policy_evaluation(env, policy, gamma=1.0, theta=0.00001):
    """
    Sutton & Barto http://incompleteideas.net/book/the-book.html

    env: openAI env
    policy: [nS, nA] shaped matrix representing the policy.
    """
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            next_state_v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    next_state_v += action_prob * prob * (reward + gamma * V[next_state])

            V[s] = next_state_v
            delta = max(delta, np.abs(v - V[s]))

        if delta < theta:
            break

    return V


def policy_improvenment(env, policy, V, gamma=1.0):
    """
    Sutton & Barto http://incompleteideas.net/book/the-book.html

    env: openAI env
    """
    policy_stable = True

    for s in range(env.nS):
        old_action = np.argmax(policy[s])

        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            next_state_value = 0
            for prob, next_state, reward, done in env.P[s][a]:
                next_state_value += prob * (reward + gamma * V[next_state])
            action_values[a] = next_state_value

        best_action = np.argmax(action_values)
        policy[s] = np.eye(env.nA)[best_action]

        if (old_action != best_action):
            policy_stable = False

    return (policy, policy_stable)


# based on https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb
def policy_iteration(env, gamma=1.0, theta=0.00001):
    """
    Sutton & Barto http://incompleteideas.net/book/the-book.html

    env: openAI env
    """
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        ## policy evaluation
        V = policy_evaluation(env, policy, gamma, theta)

        ## policy improvement
        policy, policy_stable = policy_improvenment(env, policy, V, gamma)

        if policy_stable:
            return (policy, V)

# https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb
def epsilon_greedy_exploration(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1. probability of exploration
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA

        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)

        return A

    return policy_fn


def q_learning(env, num_episodes=10000, gamma=0.95, alpha=0.1, epsilon=0.9):
    nS = env.nS
    nA = env.nA

    Q = np.zeros((nS, nA))

    stats = {}
    stats["episode_rewards"]=np.zeros(num_episodes)
    stats["episode_lengths"]=np.zeros(num_episodes)
    stats["N"]=np.zeros((nS, nA))

    V = np.zeros((num_episodes, nS))

    for i_episode in range(num_episodes):
        # epsilon greedy exploration policy we're following
        greedy_policy = epsilon_greedy_exploration(Q, epsilon, nA)
        state = np.random.choice(np.arange(nS))

        state = env.reset()

        for t in itertools.count():
            ## Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            ## epsilon greedy exploration
            action_probs = greedy_policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            ## Take a step action A, observe R, S'
            next_state, reward, done, _ = env.step(action)

            ## update stats
            stats["episode_rewards"][i_episode] += reward
            stats["episode_lengths"][i_episode] = t
            stats["N"][state, action] += 1

            ## update
            best_next_action = np.argmax(Q[next_state])
            td_delta = reward + gamma * Q[next_state][best_next_action] - Q[state][action]
            Q[state][action] += alpha * td_delta

            state = next_state

            if done:
                break

        V[i_episode, :] = Q.max(axis=1)


    return (Q, V, stats)
