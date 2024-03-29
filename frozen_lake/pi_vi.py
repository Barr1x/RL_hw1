#! python3

import numpy as np
import matplotlib.pyplot as plt
import gymnasium

import lake_info



def value_func_to_policy(env, gamma, value_func):
    '''
    Outputs a policy given a value function.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute the policy for.
    gamma: float
        Discount factor, must be in range [0, 1).
    value_func: np.ndarray
        The current value function estimate.

    Returns
    -------
    np.ndarray
        An array of integers. Each integer is the optimal action to take in
        that state according to the environment dynamics and the given value
        function.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    # BEGIN STUDENT SOLUTION
    actions_to_names = lake_info.actions_to_names
    for state in range(env.observation_space.n):
        compare_action = np.zeros(4)
        for action_index in range(4):
            action = actions_to_names[action_index]
            reward, terminal = reward_function_with_termination(env, state, action)
            if terminal:
                compare_action[action_index] = reward
            else:
                new_row, new_col = locate_position(env, state, action)
                compare_action[action_index] = reward + gamma * value_func[new_row*env.unwrapped.ncol + new_col]
        policy[state] = np.argmax(compare_action)
    # END STUDENT SOLUTION
    return(policy)



def evaluate_policy_sync(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION
    actions_to_names = lake_info.actions_to_names
    all_states = env.observation_space.n
    i = 0
    for i in range(max_iters):
        delta = 0
        value_func_updated = np.zeros(env.observation_space.n)
        for state in range(all_states):
            v = value_func[state]
            action = actions_to_names[policy[state]]
            row_new, col_new = locate_position(env, state, action)
            reward, end_episode = reward_function_with_termination(env, state, action)
            if end_episode:
                value_func_updated[state] = reward
            else:
                value_func_updated[state] = reward + gamma * value_func[row_new*env.unwrapped.ncol + col_new]
            delta = max(delta, abs(v - value_func_updated[state]))
        value_func = value_func_updated
        if delta < tol:
            break

    # END STUDENT SOLUTION
    return(value_func, i)



def evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Evaluates the value of a given policy by asynchronous DP.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION
    actions_to_names = lake_info.actions_to_names
    all_states = env.observation_space.n
    i = 0
    for i in range(max_iters):
        delta = 0
        for state in range(all_states):
            v = value_func[state]
            action = actions_to_names[policy[state]]
            row_new, col_new = locate_position(env, state, action)
            reward, end_episode = reward_function_with_termination(env, state, action)
            if end_episode:
                value_func[state] = reward
            else:
                value_func[state] = reward + gamma * value_func[row_new*env.unwrapped.ncol + col_new]
            delta = max(delta, abs(v - value_func[state]))
        if delta < tol:
            break
    # END STUDENT SOLUTION
    return(value_func, i)



def evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Evaluates the value of a policy. Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION
    actions_to_names = lake_info.actions_to_names
    all_states = env.observation_space.n
    i = 0
    for i in range(max_iters):
        delta = 0
        for state in np.random.permutation(all_states):
            v = value_func[state]
            action = actions_to_names[policy[state]]
            row_new, col_new = locate_position(env, state, action)
            reward, end_episode = reward_function_with_termination(env, state, action)
            if end_episode:
                value_func[state] = reward
            else:
                value_func[state] = reward + gamma * value_func[row_new*env.unwrapped.ncol + col_new]
            delta = max(delta, abs(v - value_func[state]))
        if delta < tol:
            break
    # END STUDENT SOLUTION
    return(value_func, i)



def improve_policy(env, gamma, value_func, policy):
    '''
    Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    value_func: np.ndarray
        The current value function estimate.
    policy: np.ndarray
        The policy to improve, maps states to actions.

    Returns
    -------
    (np.ndarray, bool)
        Returns the new policy and whether the policy changed.
    '''
    policy_changed = False
    # BEGIN STUDENT SOLUTION
    actions_to_names = lake_info.actions_to_names
    all_states = env.observation_space.n
    for state in range(all_states):
        old_action = policy[state]
        compare = np.zeros(4)
        for i in range(4):
            action = actions_to_names[i]
            reward, end_episode = reward_function_with_termination(env, state, action)
            if end_episode:
                compare[i] = reward
            else:
                row_new, col_new = locate_position(env, state, action)
                compare[i] = reward + gamma * value_func[row_new*env.unwrapped.ncol + col_new]
        policy[state] = np.argmax(compare)
        if old_action != policy[state]:
            policy_changed = True

    # END STUDENT SOLUTION
    return(policy, policy_changed)



def policy_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    # BEGIN STUDENT SOLUTION
    for i in range(max_iters):
        value_func, pe = evaluate_policy_sync(env, value_func, gamma, policy)
        pe_steps += pe
        policy, changed = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
        if not changed:
            break
    # END STUDENT SOLUTION
    return(policy, value_func, pi_steps, pe_steps)



def policy_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    # BEGIN STUDENT SOLUTION
    for i in range(max_iters):
        value_func, pe = evaluate_policy_async_ordered(env, value_func, gamma, policy)
        pe_steps += pe
        policy, changed = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
        if not changed:
            break
    # END STUDENT SOLUTION
    return(policy, value_func, pi_steps, pe_steps)



def policy_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    # BEGIN STUDENT SOLUTION
    for i in range(max_iters):
        value_func, pe = evaluate_policy_async_randperm(env, value_func, gamma, policy)
        pe_steps += pe
        policy, changed = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
        if not changed:
            break
    # END STUDENT SOLUTION
    return(policy, value_func, pi_steps, pe_steps)



def value_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    value_func_update = np.zeros(env.observation_space.n)
    actions_to_names = lake_info.actions_to_names
    i = 0
    # BEGIN STUDENT SOLUTION
    for i in range(max_iters):
        delta = 0
        compare_action = np.zeros(4)
        for state in range(env.observation_space.n):
            v = value_func[state]
            for action_index in range(4):
                action = actions_to_names[action_index]
                reward, terminal = reward_function_with_termination(env, state, action)
                if terminal:
                    compare_action[action_index] = reward
                else:
                    new_row, new_col = locate_position(env, state, action)
                    compare_action[action_index] = reward + gamma * value_func[new_row*env.unwrapped.ncol + new_col]
            value_func_update[state] = np.max(compare_action)
            delta = max(delta, abs(v - value_func_update[state]))
        value_func = value_func_update
        if delta < tol:
            break
    
    # END STUDENT SOLUTION
    return(value_func, i)


def value_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    value_func_update = np.zeros(env.observation_space.n)
    actions_to_names = lake_info.actions_to_names
    i = 0
    for i in range(max_iters):
        delta = 0
        for state in range(env.observation_space.n):
            v = value_func[state]
            compare_action = np.zeros(4)
            for action_index in range(4):
                action = actions_to_names[action_index]
                reward, terminal = reward_function_with_termination(env, state, action)
                if terminal:
                    compare_action[action_index] = reward
                else:
                    new_row, new_col = locate_position(env, state, action)
                    compare_action[action_index] = reward + gamma * value_func[new_row*env.unwrapped.ncol + new_col]
            value_func_update[state] = np.max(compare_action)
            delta = max(delta, abs(v - value_func_update[state]))
        value_func = value_func_update
        if delta < tol:
            break
    # END STUDENT SOLUTION
    return(value_func, i)



def value_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    value_func_update = np.zeros(env.observation_space.n)
    actions_to_names = lake_info.actions_to_names
    i = 0
    for i in range(max_iters):
        delta = 0
        for state in np.random.permutation(env.observation_space.n):
            v = value_func[state]
            compare_action = np.zeros(4)
            for action_index in range(4):
                action = actions_to_names[action_index]
                reward, terminal = reward_function_with_termination(env, state, action)
                if terminal:
                    compare_action[action_index] = reward
                else:
                    new_row, new_col = locate_position(env, state, action)
                    compare_action[action_index] = reward + gamma * value_func[new_row*env.unwrapped.ncol + new_col]
            value_func_update[state] = np.max(compare_action)
            delta = max(delta, abs(v - value_func_update[state]))
        value_func = value_func_update
        if delta < tol:
            break
    # END STUDENT SOLUTION
    return(value_func, i)



def value_iteration_async_custom(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    #sweep through the entire state space ordered by Manhattan distance
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    value_func_update = np.zeros(env.observation_space.n)
    actions_to_names = lake_info.actions_to_names
    sequence = sort_by_manhattan_distance(env)
    i = 0
    for i in range(max_iters):
        delta = 0
        for state in sequence:
            v = value_func[state]
            compare_action = np.zeros(4)
            for action_index in range(4):
                action = actions_to_names[action_index]
                reward, terminal = reward_function_with_termination(env, state, action)
                if terminal:
                    compare_action[action_index] = reward
                else:
                    new_row, new_col = locate_position(env, state, action)
                    compare_action[action_index] = reward + gamma * value_func[new_row*env.unwrapped.ncol + new_col]
            value_func_update[state] = np.max(compare_action)
            delta = max(delta, abs(v - value_func_update[state]))
        value_func = value_func_update
        if delta < tol:
            break
    # END STUDENT SOLUTION
    return(value_func, i)

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def sort_by_manhattan_distance(env):
    goal_location = (0,0)
    for i in range(env.unwrapped.nrow):
        for j in range(env.unwrapped.ncol):
            if map[i][j] == 'G':
                goal_location = (i,j)
    state_distance = []
    for i in range(env.observation_space.n):
        row = i // env.unwrapped.ncol
        col = i % env.unwrapped.ncol
        state_distance.append((i, manhattan_distance((row,col), goal_location)))
    state_distance.sort(key=lambda x: x[1])
    return [x[0] for x in state_distance]

# Here we provide some helper functions for your convinience.

def display_policy_letters(env, policy):
    '''
    Displays a policy as an array of letters.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to display the policy for.
    policy: np.ndarray
        The policy to display, maps states to actions.
    '''
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_info.actions_to_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.unwrapped.nrow, env.unwrapped.ncol)

    for row in range(env.unwrapped.nrow):
        print(''.join(policy_letters[row, :]))



def value_func_heatmap(env, value_func):
    '''
    Visualize a policy as a heatmap.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to display the policy for.
    value_func: np.ndarray
        The current value function estimate.
    '''
    fig, ax = plt.subplots(figsize=(7,6))

    # Reshape value_func to match the environment dimensions
    heatmap_data = np.reshape(value_func, [env.unwrapped.nrow, env.unwrapped.ncol])

    # Create a heatmap using Matplotlib
    cax = ax.matshow(heatmap_data, cmap='GnBu_r')

    # Set ticks and labels
    ax.set_yticks(np.arange(0, env.unwrapped.nrow))
    ax.set_xticks(np.arange(0, env.unwrapped.ncol))
    ax.set_yticklabels(np.arange(1, env.unwrapped.nrow + 1)[::-1])
    ax.set_xticklabels(np.arange(1, env.unwrapped.ncol + 1))

    # Display the colorbar
    cbar = plt.colorbar(cax)

    plt.show()



def locate_position(env, state, action):
    row = state // env.unwrapped.ncol
    col = state % env.unwrapped.ncol

    if action == "Left" and col > 0:
        col -= 1
    elif action == "Right" and col < env.unwrapped.ncol - 1:
        col += 1
    elif action == "Up" and row > 0:
        row -= 1
    elif action == "Down" and row < env.unwrapped.nrow - 1:
        row += 1

    return (row, col)

def reward_function_with_termination(env, state, action):
    
    row, col = locate_position(env, state, action)

    if map[row][col] == 'F':
        return (0, False)
    elif map[row][col] == 'H':
        return (0, True)
    elif map[row][col] == 'G':
        return (1, True)
    elif map[row][col] == 'S':
        return (0, False)
    
   
if __name__ == '__main__':
    np.random.seed(10003)
    maps = lake_info.maps
    gamma = 0.9

    for map_name, map in maps.items():
        env = gymnasium.make('FrozenLake-v1', desc=map, map_name=map_name, is_slippery=False)
        # BEGIN STUDENT SOLUTION
        # average_policy_steps = 0
        # average_value_steps = 0
        # for i in range(10):
        #     policy, value_func, pi_steps, pe_steps = policy_iteration_async_randperm(env, gamma)
        #     average_policy_steps += pi_steps
        #     average_value_steps += pe_steps

        # print("Policy for map: ", map_name)
        # print("Average Policy Iteration Steps: ", average_policy_steps/10)
        # print("Average Policy Evaluation Steps: ", average_value_steps/10)
        
        # average_policy_steps = 0
        # for i in range(10):
        #     value_func, i = value_iteration_async_randperm(env, gamma)
        #     average_policy_steps += i
        # print("Value for map: ", map_name)
        # print("Average Value Iteration Steps: ", average_policy_steps/10)

        value_func, i = value_iteration_async_custom(env, gamma)
        print("Value for map: ", map_name)
        print("Value Iteration Steps: ", i)
        value_func_heatmap(env, value_func)

        
       








        # END STUDENT SOLUTION
