"""This function runs in a batch mode multiple experiments"""

from sl_maliciousfunction import sl_maliciousfun
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

plt.rcParams.update({'font.size': 25})

agents = 15   #Total agents in the network
m_agents = 4  #Malicious agents in the network

weight_m = np.zeros((agents, agents))

"""In the following experiments we have true state=0. The network topology is a random network with:"""

"""A. highly discriminating models for the agents."""
"""Experiment 1: Attack strategy with unknown divergences (from Theorem 3 of the paper) with uniform priors."""
belief_history_logs1, weight_m, centrality, times_array, avg_belief_history_logs1 = \
    sl_maliciousfun(1, 1, 0, weight_m, agents, m_agents, 0, 0)
centrality_old=centrality
"""Experiment 2. Random attack strategy."""
belief_history_logs2, weight_m, centrality, times_array, avg_belief_history_logs2 = \
    sl_maliciousfun(1, 0, 1, weight_m, agents, m_agents, 0, 0)

"""B. less discriminating models for the agents."""
"""Experiment 3. Attack strategy with unknown divergences (from Theorem 3 of the paper) with uniform priors"""
belief_history_logs4, weight_m, centrality, times_array, avg_belief_history_logs4 = \
    sl_maliciousfun(1, 1, 1, weight_m, agents, m_agents, 1, 0)
""" Experiment 4. Random attack strategy"""
belief_history_logs5, weight_m, centrality, times_array, avg_belief_history_logs5 = \
    sl_maliciousfun(1, 0, 1, weight_m, agents, m_agents, 1, 0)

"""In the following experiments we have true state=0. The network topology is a star network with:"""
"""A. Highly discriminating models"""
"""Experiment 5: Attack strategy with unknown divergences (from Theorem 3 of the paper) with uniform priors."""
belief_history_logs_sdo, weight_m, centrality, times_array, avg_belief_history_logs7 = \
    sl_maliciousfun(4, 1, 0, weight_m, agents, m_agents, 0, 0)
"""Experiment 6: Random attack strategy."""
belief_history_logs_sdr,weight_m, centrality, times_array, avg_belief_history_logs8 = \
    sl_maliciousfun(4, 0, 1, weight_m, agents, m_agents, 0, 0)

"""B. Less discriminating models"""
"""Experiment 7: Attack strategy with unknown divergences (from Theorem 3 of the paper) with uniform priors."""
belief_history_logs7, weight_m, centrality, times_array, avg_belief_history_logs10 = \
    sl_maliciousfun(4, 1, 0, weight_m, agents,m_agents, 1, 0)
"""Experiment 8: Random attack strategy."""
belief_history_logs8, weight_m, centrality, times_array, avg_belief_history_logs11 = \
    sl_maliciousfun(4, 0, 1, weight_m, agents, m_agents, 1, 0)

"""In the following experiments we have true state=1. The network topology is a random network with:"""

"""A. Highly discriminating models"""
"""Experiment 9: Attack strategy with unknown divergences (from Theorem 3 of the paper) with uniform priors."""
belief_history_logs10, weight_m, centrality, times_array, avg_belief_history_logs13 = \
    sl_maliciousfun(1, 1, 0, weight_m, agents, m_agents, 0, 1)
"""Experiment 10: Random attack strategy."""
belief_history_logs11, weight_m, centrality, times_array, avg_belief_history_logs14 = \
    sl_maliciousfun(1, 0, 1, weight_m, agents, m_agents, 0, 1)

"""B. Less discriminating models"""
""""Experiment 11: Attack strategy with unknown divergences (from Theorem 3 of the paper) with uniform priors."""
belief_history_logs13, weight_m, centrality, times_array, avg_belief_history_logs16 = \
    sl_maliciousfun(1, 1, 1, weight_m, agents, m_agents, 1, 1)
""""Experiment 12: Random attack strategy."""
belief_history_logs14, weight_m, centrality, times_array, avg_belief_history_logs17 = \
    sl_maliciousfun(1, 0, 1, weight_m, agents, m_agents, 1, 1)

"""In the following experiments we have true state=1. The network topology is a star network with:"""

"""A. Highly discriminating models"""
"""Experiment 13: Attack strategy with unknown divergences (from Theorem 3 of the paper) with uniform priors."""
belief_history_logs_sdo2, weight_m, centrality, times_array, avg_belief_history_logs19 = \
    sl_maliciousfun(4, 1, 0, weight_m, agents, m_agents, 0, 1)
"""Experiment 14: Random attack strategy."""
belief_history_logs_sdr2, weight_m, centrality, times_array, avg_belief_history_logs20 = \
    sl_maliciousfun(4, 0, 1, weight_m, agents, m_agents, 0, 1)

"""B. Less discriminating models"""
"""Experiment 15: Attack strategy with unknown divergences (from Theorem 3 of the paper) with uniform priors."""
belief_history_logs16, weight_m, centrality, times_array, avg_belief_history_logs22 = \
    sl_maliciousfun(4, 1, 0, weight_m, agents, m_agents, 1, 1)
"""Experiment 16: Random attack strategy."""
belief_history_logs17, weight_m, centrality, times_array, avg_belief_history_logs23 = \
    sl_maliciousfun(4, 0, 1, weight_m, agents, m_agents, 1, 1)

"""The following figure depicts the average (across agents) belief evolution of the agents."""
fig = plt.figure(figsize=(5, 3))

plt.subplot(1, 2, 1)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.plot(times_array.T, avg_belief_history_logs4[:, 0], linewidth=2, color='r')
plt.plot(times_array.T, avg_belief_history_logs5[:, 0], linewidth=2, color='b')

plt.plot(times_array.T, avg_belief_history_logs16[:, 1], '--', linewidth=2, color='r')
plt.plot(times_array.T, avg_belief_history_logs17[:, 1], '--', linewidth=2, color='b')

plt.title('Random network', fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.ylabel(r'$\bar{\mu}_i(\theta^{\star})$', fontsize=15)
plt.ylim([-0.1, 1.1])
plt.grid()

plt.subplot(1, 2, 2)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.plot(times_array.T, avg_belief_history_logs10[:,0], linewidth=2, color='r')
plt.plot(times_array.T, avg_belief_history_logs11[:,0], linewidth=2, color='b')

plt.plot(times_array.T, avg_belief_history_logs22[:, 1], '--', linewidth=2, color='r')
plt.plot(times_array.T, avg_belief_history_logs23[:, 1], '--', linewidth=2, color='b')

plt.title('Star network', fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.ylabel(r'$\bar{\mu}_{k,i}(\theta^{\star})$', fontsize=15)
plt.grid()
plt.ylim([-0.1, 1.1])

labels=[r'ASUD - $\theta^{\star}=\theta_1$',r'RAS - $\theta^{\star}=\theta_1$'\
    ,r'ASUD - $\theta^{\star}=\theta_2$',\
        r'RAS - $\theta^{\star}=\theta_2$']

plt.legend(labels, loc="lower center", bbox_to_anchor=(-0.25,-0.9), ncol=2, fontsize=12)

fig.subplots_adjust(bottom=0.42, wspace=0.4)
plt.grid(which = 'minor')

plt.savefig('Fig_2_time_50asud.png')
plt.savefig('Fig_2_time_50asud.pdf')

plt.show()

"""The following figure depicts the average (across agents) belief evolution of the agents."""
fig = plt.figure(figsize=(5, 3))

plt.subplot(1,2,1)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.plot(times_array.T,avg_belief_history_logs1[:, 0], linewidth=2, color='r')#label=r'Optimal attack strategy - $\theta=\theta_1$'
plt.plot(times_array.T,avg_belief_history_logs2[:, 0], linewidth=2, color='b')#label=r'Random attack strategy - $\theta=\theta_1$'

plt.plot(times_array.T,avg_belief_history_logs13[:, 1], '--',linewidth=2,color='r')#label=r'Optimal attack strategy - $\theta=\theta_2$',linewidth=2.5,color='r')
plt.plot(times_array.T,avg_belief_history_logs14[:, 1], '--',linewidth=2,color='b')#label=r'Random attack strategy - $\theta=\theta_2$',linewidth=2.5,color='b')

plt.title('Random network', fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.ylabel(r'$\bar{\mu}_{k,i}(\theta^{\star})$',fontsize=15)
plt.grid()
plt.ylim([-0.1,1.1])
plt.grid(which = 'minor')

plt.subplot(1,2,2)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.plot(times_array.T,avg_belief_history_logs7[:,0],linewidth=2,color='r')#label=r'Optimal attack strategy - $\theta=\theta_1$',linewidth=2.5,color='r')
plt.plot(times_array.T,avg_belief_history_logs8[:,0],linewidth=2,color='b')#label=r'Random attack strategy - $\theta=\theta_1$',linewidth=2.5,color='b')

plt.plot(times_array.T,avg_belief_history_logs19[:,1],'--',linewidth=2,color='r')#label=r'Optimal attack strategy - $\theta=\theta_2$',linewidth=2.5,color='r')
plt.plot(times_array.T,avg_belief_history_logs20[:,1],'--',linewidth=2,color='b')#label=r'Random attack strategy - $\theta=\theta_2$',linewidth=2.5,color='b')
plt.grid(which = 'minor')
plt.title('Star network',fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.ylabel(r'$\bar{\mu}_{k,i}(\theta^{\star})$',fontsize=15)
plt.grid()
plt.ylim([-0.1, 1.1])

labels=[r'ASUD - $\theta^{\star}=\theta_1$',r'RAS - $\theta^{\star}=\theta_1$',\
        r'ASUD - $\theta^{\star}=\theta_2$',r'RAS - $\theta^{\star}=\theta_2$']

plt.legend(labels, loc="lower center", bbox_to_anchor=(-0.25, -0.9), ncol=2, fontsize=12)

fig.subplots_adjust(bottom=0.42,wspace=0.4)

plt.savefig('Fig_1_time_50asud.png')
plt.savefig('Fig_1_time_50asud.pdf')

plt.show()