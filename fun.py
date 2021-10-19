import numpy as np
from numpy.random import binomial, multinomial


def create_network(agents, option):
    """Creates an undirected graph (with self-loops) constructed according to option.

    input:  agents: number of agents
            option: option of construction

    returns: adj: the network's adjacency matrix (a matrix of 1 and 0. 1 means agents are connect, 0 means they are not.
    """

    adj_matrix = np.zeros((agents, agents))         #Adjacency matrix of the network

    if option == 1:
        """Erdos-Renyi construction"""
        adj_matrix = np.random.choice([0.0, 1.0], size=(agents, agents), p=[0.7, 0.3])
        adj_matrix = adj_matrix + np.eye(agents)
        adj_matrix = (adj_matrix > 0) * 1.0
    elif option == 2:
        """Fully connected"""
        adj_matrix = np.ones((agents, agents))
    elif option == 4:
        """Star network with self-loops"""
        adj_matrix = np.zeros((agents, agents))
        center_node = 0
        for n in range(agents):
            adj_matrix[n, center_node] = 1
            adj_matrix[center_node, n] = 1
            adj_matrix[n, n] = 1
    else:
        print("Invalid selection of the matrix construction.")
        exit()
    return adj_matrix


def combination_weights(adj, agents):
    """Creates the combination weight of the network (only uniform combination weights).

        input:      adj: adjacency matrix
                    option: option of construction
                    agents: number of agents in he network

        returns:    weights: the network's adjacency matrix
                    centrality: The Perron eigenvector corresponding to eigenvalue 1.
    """

    weight = np.zeros((agents, agents))
    for k in range(agents):
        temp_sum = sum(adj[:, k])
        for nei in range(agents):
            if adj[nei, k] == 1:
                weight[nei, k] = 1 / temp_sum
    e_values, e_vectors = np.linalg.eig(weight)
    e_index = np.argmax(e_values)
    centrality = e_vectors[:, e_index] / e_vectors[:, e_index].sum()
    strongly_connected = np.all(centrality > 0)

    if strongly_connected != True:
        print("Not strongly connected network. Run the code again.")
        exit()

    return weight, centrality


def generator(s, players, obs_mat):
    assert players is not None, "Choose player in next belief"

    o = np.zeros((players, 1), np.int)  # New
    for player in range(players):
        temp = multinomial(1, obs_mat[player, s, :])
        temp = int(np.nonzero(temp)[0])
        o[player, 0] = temp

    return o


def create_observation_models(agents, states, obs, strategy):
    """
    Create the likelihood functions of the agents.

    Input:  agents: number of agents
            states: number of states
            obs: number of observations
            strategy: option of construction

    Return: Agents' likelihood functions
    """
    obs_matrix = np.random.rand(agents, states, obs)
    for agent in range(agents):
        for row in range(states):
            obs_matrix[agent, row, :] = obs_matrix[agent, row, :] / (sum(obs_matrix[agent, row, :]))  # does it work?
    if strategy == 0: #uniform(same for all agents) manual, discriminative models (between the two states)
        for agent in range(agents):
            obs_matrix[agent, 0, 0] = 0.95
            obs_matrix[agent, 0, 1] = 1-obs_matrix[agent, 0, 0]
            obs_matrix[agent, 1, 0] = 0.05
            obs_matrix[agent, 1, 1] = 1-obs_matrix[agent, 1, 0]
    elif strategy == 1:#uniform(same for all agents) manual, less discriminative models
        for agent in range(agents):
            obs_matrix[agent, 0, 0] = 0.8
            obs_matrix[agent, 0, 1] = 1-obs_matrix[agent, 0, 0]
            obs_matrix[agent, 1, 0] = 0.2
            obs_matrix[agent, 1, 1] = 1-obs_matrix[agent, 1, 0]
    return obs_matrix


def calculate_network_divergence(nagents, magents, state1, state2, obs_mat, centrality):
    """Calculate the divergence of the normal sub-network."""
    nnd = 0
    for agent in range(nagents):
        if agent > (magents - 1):#normal agents
            kl = kl_div(obs_mat[agent, state1, :], obs_mat[agent, state2, :], 2)#last argument:number of observations
            nnd = nnd+centrality[agent]*kl
    return nnd


def create_distortion_functions(agents, states, obs, strategy, prior, obs_mat, nnd0, nnd1, centrality):
    """
    Create the distorted likelihood functions according to selected strategy

    return: f_m, which represents the distorted PMFs of the agents
    """
    f_m = np.zeros((agents, states, obs))
    if strategy == 0:   #random attack strategy
        f_m = np.random.rand(agents, states, obs)
        for agent in range(agents):
            for row in range(states):
                f_m[agent, row, :] = f_m[agent, row, :] / (
                    sum(f_m[agent, row, :]))
    elif strategy==1:   #optimal expected cost
        epsilon = 0.001  # Minimum admissible probability mass
        for agent in range(agents):
            # Check if the case is the uniform or the non uniform
            neg_coefficients=0
            pos_coefficients=0
            min_zeta = 0     #observation that yields the minimum coefficient for true state 0
            max_zeta = 0
            sum_pos = 0
            sum_neg = 0
            for zeta in range(obs):
                if (prior[0, 0]*obs_mat[agent, 0, zeta]-prior[1, 0]*obs_mat[agent, 1, zeta]) < \
                        (prior[0, 0]*obs_mat[agent, 0, min_zeta]-prior[1, 0]*obs_mat[agent, 1, min_zeta]):
                    min_zeta = zeta
                if (prior[0, 0]*obs_mat[agent, 0, zeta]-prior[1, 0]*obs_mat[agent, 1, zeta]) > \
                        (prior[0, 0]*obs_mat[agent, 0, max_zeta]-prior[1, 0]*obs_mat[agent, 1, max_zeta]):
                    max_zeta = zeta
                if (prior[0, 0]*obs_mat[agent, 0, zeta]-prior[1, 0]*obs_mat[agent, 1, zeta]) >= 0:
                    pos_coefficients=pos_coefficients+1
                    sum_pos=sum_pos+(prior[0,0]*obs_mat[agent, 0, zeta]-prior[1, 0]*obs_mat[agent, 1, zeta])
                else:
                    neg_coefficients = neg_coefficients+1
                    sum_neg = sum_neg+(prior[0, 0]*obs_mat[agent, 0, zeta]-prior[1, 0]*obs_mat[agent, 1, zeta])
            if neg_coefficients == 0:#state 0 is more likely for every observation
                uniform = 0
                for zeta in range(obs):
                    if zeta == min_zeta:
                        f_m[agent, 0, zeta] = 1-(obs-1)*epsilon
                    else:
                        f_m[agent, 0, zeta] = epsilon
                    f_m[agent, 1, zeta] = ((prior[0, 0]*obs_mat[agent, 0, zeta] - prior[1, 0]*obs_mat[agent, 1, zeta]))/(sum_pos)
            elif pos_coefficients==0:#state 1 is more likely for every observation
                for zeta in range(obs):
                    if zeta==max_zeta:
                        f_m[agent,1,zeta]=1-(obs-1)*epsilon
                    else:
                        f_m[agent, 1, zeta] =epsilon
                    f_m[agent,0,zeta]=(prior[0,0]*obs_mat[agent, 0, zeta]-prior[1, 0]*obs_mat[agent, 1, zeta])/(sum_neg)
            else:#nonuniform scenario, some observations more likely from state 0, some from state 1
                for zeta in range(obs):
                    if (prior[0,0]*obs_mat[agent,0,zeta]-prior[1,0]*obs_mat[agent,1,zeta])>=0:
                        f_m[agent, 0, zeta]=epsilon
                        f_m[agent, 1, zeta] =(prior[0,0]*obs_mat[agent,0,zeta]-prior[1,0]*obs_mat[agent,1,zeta])*\
                                             (1-epsilon*sum_pos)/(sum_pos)
                    else:
                        f_m[agent, 1, zeta] = epsilon
                        f_m[agent, 0, zeta] = (prior[0, 0] * obs_mat[agent, 0, zeta] - prior[1, 0] * obs_mat[agent, 1, zeta]) * \
                                              (1 - epsilon * sum_neg) / (sum_neg)
    elif strategy == 2:     #flip strategy
        for agent in range(agents):
            f_m[agent, 0, :]=obs_mat[agent, 1, :]
            f_m[agent, 1, :] = obs_mat[agent, 0, :]
        print("f_m",f_m)
    elif strategy == 3:     #no attack
        f_m = obs_mat
    elif strategy == 4:     #manual
        print(agents, states, obs)
        f_m = np.random.rand(agents, states, obs)
        f_m[0,0,0] = 0   #(states,observations)
        f_m[0,0,1] = 1-f_m[0,0,0]
        f_m[0,1,0] = 1
        f_m[0,1,1] = 1-f_m[0,1,0]
    elif strategy == 5:#optimal with known divergences
        for agent in range(agents):
            print("inside strategy")
            print(obs_mat)
            print(centrality)
            print("nnd0", nnd0)
            print("nnd1", nnd1)
            det=obs_mat[agent,0,0]*obs_mat[agent,1,1]-obs_mat[agent,1,0]*obs_mat[agent,0,1]
            #Intersection point
            x1 = -(obs_mat[agent,1,1] * nnd0 + obs_mat[agent,0,1] * nnd1) / (centrality[agent]*(-obs_mat[agent,1,1] * \
                obs_mat[agent,0,0] + obs_mat[agent,0,1] *obs_mat[agent,1,0]))#agent 0:one agent is enough
            x2 = (obs_mat[agent,1,0] * nnd0 + obs_mat[agent,0,0] * nnd1) / (centrality[agent]*(-obs_mat[agent,1,1] * \
                obs_mat[agent,0,0] + obs_mat[agent,0,1] * obs_mat[agent,1,0]))# (q21*L1+q11*L2)/(q12*q21-q11*q22)+0.01
            print("intersection point",x1,x2)
            """Pick a point satisfying inequalities"""
            if x1<0:
                x1_point=x1-0.5
            else:
                x1_point = x1+0.5
            x2_line0 = (nnd0 - centrality[agent] * obs_mat[agent, 0, 0] * x1_point) / (
                    centrality[agent] * obs_mat[agent, 0, 1])  # line corresponding to true state=0
            x2_line1 = -(nnd1 + centrality[agent] * obs_mat[agent, 1, 0] * x1_point) / (
                    centrality[agent] * obs_mat[agent, 1, 1])  # line corresponding to true state=1
            print("values of x2 lines",x2_line0,x2_line1)
            x2_point = (x2_line0 + x2_line1) / 2
            if det<0:
                e2 = (1-np.exp(x1_point)) / (-np.exp(x1_point) + np.exp(x2_point))
                e1 = np.exp(x1_point)- np.exp(x1_point)*(((np.exp(-x1_point)) - 1) / (-1 + (np.exp(x2_point - x1_point))))
                c1 = nnd0 +centrality[agent]*(- obs_mat[agent,0,0] * x1_point - obs_mat[agent,0,1] * x2_point)  # condition 1
                c2 = nnd1 + centrality[agent]*(obs_mat[agent,1,0] * x1_point + obs_mat[agent,1,1] * x2_point)  # condition 2
                print(c1, c2, x1_point, x2_point, e1, e2)
                if (e1 < 0) or (e1 > 1) or (e2 < 0) or (e2 > 1):
                    print(np.exp(x1_point))
                    print(np.exp(x2_point))
                    print("e1,e2>1!")
                    exit()
                x1_array = np.arange(20, 100, 0.1)
                x2_f1 = np.zeros((len(x1_array), 1))
                x2_f2 = np.zeros((len(x1_array), 1))
                print(obs_mat[0, :, :])

                f_m[agent, 0, 0] = 1 - e2
                f_m[agent, 0, 1] = e2
                f_m[agent, 1, 0] = e1
                f_m[agent, 1, 1] = 1 - e1
            else:
                e2_alt=(1-np.exp(x2_point)) / (-np.exp(x2_point) + np.exp(x1_point))
                e1_alt = np.exp(x2_point) - np.exp(x2_point) * (
                            (-(np.exp(-x2_point)) + 1) / (-1 + (np.exp(x1_point - x2_point))))
                c1 = nnd0 + centrality[agent] * (
                            - obs_mat[agent, 0, 0] * x1_point - obs_mat[agent, 0, 1] * x2_point)  # condition 1
                c2 = nnd1 + centrality[agent] * (
                            obs_mat[agent, 1, 0] * x1_point + obs_mat[agent, 1, 1] * x2_point)  # condition 2
                print(c1, c2, x1_point, x2_point, e1_alt, e2_alt)
                if (e1_alt < 0) or (e1_alt > 1) or (e2_alt < 0) or (e2_alt > 1):
                    print(np.exp(x1_point))
                    print(np.exp(x2_point))
                    print("e1,e2>1!")
                    exit()
                x1_array = np.arange(20, 100, 0.1)
                x2_f1 = np.zeros((len(x1_array), 1))
                x2_f2 = np.zeros((len(x1_array), 1))
                print(obs_mat[0, :, :])

                f_m[agent, 0, 0] = e2_alt
                f_m[agent, 0, 1] = 1-e2_alt
                f_m[agent, 1, 0] = 1-e1_alt
                f_m[agent, 1, 1] = e1_alt
    return f_m


def next_belief_dif_malicious(belief, obs, obs_matrix,player,states,players,f_m,attack_str):
    """
    Calclulates next belief for a malicious agent:
        :param belief:
        :param obs:
        :param st_matrix:
        :param obs_matrix:
        :param player:
        :param states:
        :param players:
        :param f_m:
            attack_str: 0:random attack strategy, 1:optimal expected cost
        :return:
    """
    new_belief = np.zeros((states, 1))
    for s in range(states):
        new_belief[s] = f_m[player, s, obs[player, 0]]*belief[s]
    # Calculate denominator
    denominator = 0
    for sn in range(states):
        denominator = denominator + new_belief[sn]
    new_belief = new_belief / denominator

    return new_belief[:, 0]


def next_belief_dif(belief, obs, obs_matrix, player, states):
    """
    Calculates next belief:
        arguments - current belief, observation obtained by me, observation obtained by adversary, who_am_i?
    """

    new_belief = np.zeros((states, 1))
    for s in range(states):
        new_belief[s] = obs_matrix[player, s, obs[player, 0]]*belief[s]
    #Calculate denominator
    denominator = 0
    for sn in range(states):
        denominator = denominator+new_belief[sn]
        #Check
    for s in range(states):
        if new_belief[s, 0] < 0 or new_belief[s, 0] > 1:
            print("Here next_belief_dif.", s, player, new_belief[0, 0], new_belief[1, 0])
            exit()
    new_belief = new_belief/denominator
    return new_belief[:, 0]


def kl_div(dis1, dis2, observations):
    """ Input: two distributions
        Output: kl divergence of dis1 over dis2"""
    kl = 0
    for obs in range(observations):
        kl = kl+dis1[obs]*np.log(dis1[obs]/dis2[obs])
    return kl
