# social_learning_under_inferential_attacks_ICASSP2021

This is the code that corresponds to the simulations performed for the paper:

K. Ntemos, V. Bordignon, S. Vlaski and A. H. Sayed, “Social Learning Under Inferential Attacks,” in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 5479-5483, doi: 10.1109/ICASSP39728.2021.9413642.

Description of the files

batch_file.py: This is the function that runs the various experiments and outputs the respective figures that appear in the paper.

NOTE: the final results may vary with the results presented in the paper due to the randomness in the network construction. This affects the agents’ centrality and as a result the results might differ slightly. For example, maybe more time instants are needed for the beliefs to converge to their limiting values. This can be easily adjusted by increasing the value of the variable times in sl_maliciousfunction.py.

sl_maliciousfunction.py: This file runs an experiment for social learning with adversaries.

fun.py: This file contains various auxiliary functions.

Please note that the code is not generally perfected for performance, but is rather meant to illustrate certain results from the paper. The code is provided as-is without guarantees.

Author: Konstantinos Ntemos.
