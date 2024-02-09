#! python3

import numpy as np
import matplotlib.pyplot as plt



# Use the run exploration algorithm we have provided to get you
# started, this returns the expected rewards after running an exploration 
# algorithm in the K-Armed Bandits problem. We have already specified a number
# of parameters specific to the 10-armed testbed for guidance.



def runExplorationAlgorithm(explorationAlgorithm, param, iters):
    cumulativeRewards = []
    for i in range(iters):
        # number of time steps
        t = 1000
        # number of arms, 10 in this instance
        k = 10
        # mean reward across each of the K arms
        # sample the actual rewards from a normal distribution with mean of meanRewards and standard deviation of 1
        meanRewards = np.random.normal(1,1,k)
        # counts for each arm
        n = np.zeros(k)
        # extract expected rewards by running specified exploration algorithm with the parameters above
        # param is the different, specific parameter for each exploration algorithm
        # this would be epsilon for epsilon greedy, initial values for optimistic intialization, c for UCB, and temperature for Boltmann
        currentRewards = explorationAlgorithm(param, t, k, meanRewards, n)
        cumulativeRewards.append(currentRewards)
    # calculate average rewards across each iteration to produce expected rewards
    expectedRewards = np.mean(cumulativeRewards, axis=0)
    return expectedRewards



def epsilonGreedyExploration(epsilon, steps, k, meanRewards, n):
    # TODO implement the epsilong greedy algorithm over all steps and return
    # the expected rewards across all steps
    expectedRewards = np.zeros(steps)

    # BEGIN STUDENT SOLUTION
    actionsValue = np.zeros(k)
    actionsNumber = np.zeros(k)
    actionPolicy = np.zeros(k)

    for i in range(steps):
        if (np.random.rand() < epsilon):
            action = np.random.randint(0, k)
        else:
            action = np.argmax(actionsValue)

        actualReward = np.random.normal(meanRewards[action], 1)

        actionsNumber[action] += 1
        actionsValue[action] += (actualReward - actionsValue[action]) / (actionsNumber[action])

        for j in range (k):
            if (j == action):
                actionPolicy[j] = (1 - epsilon) + (epsilon)/k
            else:
                actionPolicy[j] = (epsilon)/k

        expectedRewards[i] = np.dot(meanRewards, actionPolicy)
    # END STUDENT SOLUTION
    return(expectedRewards)



def optimisticInitialization(value, steps, k, meanRewards, n):
    # TODO implement the optimistic initializaiton algorithm over all steps and
    # return the expected rewards across all steps
    expectedRewards = np.zeros(steps)

    # BEGIN STUDENT SOLUTION
    actionsValue = np.full(k, value)
    actionsNumber = np.zeros(k)

    for i in range(steps):
        action = np.argmax(actionsValue)
        actualReward = np.random.normal(meanRewards[action], 1)
        actionsNumber[action] = actionsNumber[action] + 1
        actionsValue[action] = actionsValue[action] + (actualReward - actionsValue[action]) / (actionsNumber[action])

        expectedRewards[i] = actualReward
    # END STUDENT SOLUTION
    return(expectedRewards)



def ucbExploration(c, steps, k, meanRewards, n):
    # TODO implement the UCB exploration algorithm over all steps and return the
    # expected rewards across all steps, remember to pull all arms initially
    expectedRewards = np.zeros(steps)

    # BEGIN STUDENT SOLUTION
    actionsValue = np.zeros(k)
    actionsNumber = np.zeros(k)

    for i in range(steps):
        if (np.all(actionsNumber)):
            action = np.argmax(actionsValue + c * np.sqrt(np.log(i) / actionsNumber))
        else:
            action = np.argmin(actionsNumber)

        actualReward = np.random.normal(meanRewards[action], 1)

        actionsNumber[action] += 1
        actionsValue[action] += (actualReward - actionsValue[action]) / (actionsNumber[action])

        expectedRewards[i] = actualReward
    # END STUDENT SOLUTION
    return(expectedRewards)



def boltzmannExploration(temperature, steps, k, meanRewards, n):
    # TODO implement the Boltzmann Exploration algorithm over all steps and
    # return the expected rewards across all steps
    expectedRewards = np.zeros(steps)

    # BEGIN STUDENT SOLUTION
    actionsValue = np.zeros(k)
    actionsNumber = np.zeros(k)

    for i in range(steps):
        actionPolicy = np.exp(actionsValue*temperature) / np.sum(np.exp(actionsValue*temperature))
        action = np.random.choice(k, p=actionPolicy)

        actualReward = np.random.normal(meanRewards[action], 1)

        actionsNumber[action] += 1
        actionsValue[action] += (actualReward - actionsValue[action]) / (actionsNumber[action])

        expectedRewards[i] = np.dot(meanRewards, actionPolicy)
    
    # END STUDENT SOLUTION
    return(expectedRewards)



# plot template
def plotAlgorithms(alg_param_list):
    # TODO given a list of (algorithm, parameter) tuples, make a graph that
    # plots the expectedRewards of running that algorithm with those parameters
    # iters times using runExplorationAlgorithm plot all data on the same plot
    # include correct labels on your plot
    iters = 1000
    alg_to_name = {epsilonGreedyExploration : 'Epsilon Greedy Exploration',
                   optimisticInitialization : 'Optimistic Initialization',
                   ucbExploration: 'UCB Exploration',
                   boltzmannExploration: 'Boltzmann Exploration'}
    # BEGIN STUDENT SOLUTION
    for alg, param in alg_param_list:
        rewards = runExplorationAlgorithm(alg, param, iters)
        plt.plot(rewards, label=f'{alg_to_name[alg]} ({param})')

    plt.xlabel('Time Steps')
    plt.ylabel('Expected Rewards')
    plt.legend()
    plt.show()
    # END STUDENT SOLUTION
    pass



if __name__ == '__main__':
    # TODO call plotAlgorithms here to plot your algorithms
    np.random.seed(10003)

    # BEGIN STUDENT SOLUTION
    plotAlgorithms([(epsilonGreedyExploration, 0.1), (optimisticInitialization, 1), (ucbExploration, 1), (boltzmannExploration, 3)])
    # END STUDENT SOLUTION
    pass
