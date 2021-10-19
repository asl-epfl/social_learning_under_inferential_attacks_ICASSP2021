import numpy as np
from numpy.random import binomial
from fun import generator
from fun import next_belief_dif
from fun import create_network
from fun import combination_weights
from fun import create_distortion_functions
from fun import next_belief_dif_malicious
from fun import create_observation_models
from fun import calculate_network_divergence
import decimal


def sl_maliciousfun(network_type, malicious_strategy, default_network_flag,
            default_network, agents, m_agents, opt_observations, true_state):
    """This function executes the social learning with malicious agents able to launch inferential attacks.
    Input:  network_type: random or star
            malicious_strategy:  : Attack strategy with known divergences
                                 : Attack strategy with unknown divergences
            default_network_flag :
            default_network
            agents: number of total agents in the netework
            m_agents: number of malicious agents in the network
            opt_observations:
            true_state:
    Output: belief_history_logs, weight, centrality, np.arange(1, times), avg_belief_history_logs
    """
    states = 2           #The number of possible states. Note that the algorithm is written for the binary hypothesis setup, meaning that it should always be states=2.
    observations = 2     #Number of observations per agent
    times = 50           #Simulation periods

    s_0 = true_state     #The true hypothesis/state
    if s_0 == 1:
        s_alt = 0
    else:
        s_alt = 1

    if default_network_flag==0:
        """In this case the network is not given as input to the function."""
        """Create the network and the combination matrix."""
        adj_matrix = create_network(agents, network_type)#Second argument is the option. 0:Manual, 1:Random, 2:Fully connected, 3:Erdos-Renyi,4:star
        weight, centrality = combination_weights(adj_matrix, agents) #row to vector
    else:
        """In this case the network is given as input."""
        weight = default_network
        e_values, e_vectors = np.linalg.eig(weight)
        e_index = np.argmax(e_values)
        centrality = e_vectors[:, e_index] / e_vectors[:, e_index].sum()
        strongly_connected = np.all(centrality > 0)

        if strongly_connected != True:
            print("Not strongly connected network. Run the code again.")
            exit()

    """Initialization"""
    initial_belief_vector = (1/states)*np.ones((states, 1))#Initial belief vector - Set to uniform
    belief_vector_dif_logs = np.zeros((states, agents))
    belief_vector_logs_combine = np.zeros((states, agents))
    belief_history_logs = np.zeros((times, agents, states))

    for a in range(agents):
        for s in range(states):
            belief_vector_dif_logs[s, a] = decimal.Decimal(initial_belief_vector[s, 0])
            belief_vector_logs_combine[s, a] = decimal.Decimal(initial_belief_vector[s, 0])

    """ Create the agents' likelihood functions and adversaries'
    fake likelihood functions."""
    obs_matrix = create_observation_models(agents, states, observations, opt_observations)
    nnd0 = calculate_network_divergence(agents, m_agents, 0, 1, obs_matrix, centrality)  # true state=0
    nnd1 = calculate_network_divergence(agents, m_agents, 1, 0, obs_matrix, centrality)  # true state=1
    f_m = create_distortion_functions(m_agents, states, observations, malicious_strategy,
                                      initial_belief_vector, obs_matrix, nnd0, nnd1, centrality)
    s = s_0
    """Simulations start"""
    for t in range(times):
        o = generator(s, agents, obs_matrix)    #Create agents' observations.

        """Adaptation step. Agents update their beliefs based on the received observations
        according to Bayes' rule."""
        for agent in range(agents):
            if agent <= (m_agents-1):  #malicious agents
                belief_vector_dif_logs[:, agent] = next_belief_dif_malicious(belief_vector_logs_combine[:, agent], o,
                                                                             obs_matrix, agent, states, m_agents, f_m,
                                                                             malicious_strategy)
            else:   #normal agents
                for s_temp in range(states):
                    if belief_vector_dif_logs[s_temp, agent] < 0:
                        print("Here logs!!!!!!", belief_vector_dif_logs[s_temp, agent], t)
                belief_vector_dif_logs[:, agent] = \
                    next_belief_dif(belief_vector_logs_combine[:, agent], o, obs_matrix, agent, states)
        """Combination step. Agents update their beliefs based on the received beliefs from their neighbors."""
        for agent in range(agents):
            belief_history_logs[t, agent, :] = belief_vector_logs_combine[:, agent]
            belief_temp3 = np.ones((states, 1))
            for neigh in range(agents):
                for s_temp in range(states):
                    belief_temp3[s_temp, 0] = belief_temp3[s_temp, 0] * \
                                                  (belief_vector_dif_logs[s_temp, neigh] ** weight[neigh, agent])

            # Calculate denominator
            den_1 = 0
            for st_temp in range(states):
                den_1 = den_1 + belief_temp3[st_temp, 0]
            for st_temp in range(states):
                belief_temp3[st_temp, 0] = belief_temp3[st_temp, 0] / den_1
            belief_vector_logs_combine[:, agent] = belief_temp3[:, 0]

    avg_belief_history_logs=np.zeros((times, states))
    for state in range(states):
        for agent in range(agents):
            avg_belief_history_logs[:, state]=avg_belief_history_logs[:, state]+belief_history_logs[:, agent, state]
    avg_belief_history_logs=avg_belief_history_logs/agents

    return belief_history_logs, weight, centrality, np.arange(1, times+1), avg_belief_history_logs
