#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/04/02

@author: frederic

Some sanity tests

Here is the console output

    = = = = = = = = = = = = = = = = = = = =
    Mine of depth 4
    Plane x,z view
    [[-0.814  0.559  0.175  0.212 -1.231]
     [ 0.637 -0.234 -0.284  0.088  1.558]
     [ 1.824 -0.366  0.026  0.304 -0.467]
     [-0.563  0.07  -0.316  0.604 -0.371]]
    -------------- DP computations --------------
    Cache DP function before first call :  CacheInfo(hits=0, misses=0, maxsize=None, currsize=0)
    Cache DP function after all calls :   CacheInfo(hits=432, misses=259, maxsize=None, currsize=259)
    DP Best payoff  2.9570000000000003
    DP Best final state  (3, 2, 3, 4, 3)
    DP action list  ((0,), (1,), (0,), (2,), (1,), (0,), (3,), (2,), (4,), (3,), (2,), (4,), (3,), (4,), (3,))
    DP Computation took 0.015289068222045898 seconds

    -------------- BB computations --------------
    Cache BB optimistic_value function before first call :  CacheInfo(hits=0, misses=0, maxsize=None, currsize=0)
    Cache BB optimistic_value function after all calls:  CacheInfo(hits=8, misses=215, maxsize=None, currsize=215)
    BB Best payoff  2.9570000000000003
    BB Best final state  (3, 2, 3, 4, 3)
    BB action list  [(0,), (1,), (2,), (3,), (4,), (0,), (1,), (2,), (3,), (4,), (0,), (2,), (3,), (4,), (3,)]
    BB Computation took 0.027152538299560547 seconds
    = = = = = = = = = = = = = = = = = = = =
    Mine of depth 5
    Level by level x,y slices
    level 0
    [[ 0.455  0.049  2.38   0.515]
     [ 0.801 -0.09  -1.815  0.708]
     [-0.857 -0.876 -1.936  0.316]]
    level 1
    [[ 0.579  1.311 -1.404 -0.236]
     [ 0.072 -1.191 -0.839 -0.227]
     [ 0.309  1.188 -3.055  0.97 ]]
    level 2
    [[-0.54  -0.061  1.518 -0.466]
     [-2.183 -1.083  0.457  0.874]
     [-1.623 -0.16  -0.535  1.097]]
    level 3
    [[-0.995  0.185 -0.856 -1.241]
     [ 0.858  0.78  -1.029  1.563]
     [ 0.364  0.888 -1.561  0.234]]
    level 4
    [[-0.771 -1.959  0.658 -0.354]
     [-1.504 -0.763  0.915 -2.284]
     [ 0.097 -0.546 -1.992 -0.296]]
    -------------- DP computations --------------
    Cache DP function before first call :  CacheInfo(hits=0, misses=0, maxsize=None, currsize=0)
    Cache DP function after all calls :   CacheInfo(hits=176863, misses=39098, maxsize=None, currsize=39098)
    DP Best payoff  5.713
    DP Best final state  ((2, 1, 1, 1), (1, 1, 0, 1), (0, 0, 0, 1))
    DP action list  ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (0, 0), (1, 3), (2, 3))
    DP Computation took 6.34877872467041 seconds

    -------------- BB computations --------------
    Cache BB optimistic_value function before first call :  CacheInfo(hits=0, misses=0, maxsize=None, currsize=0)
    Cache BB optimistic_value function after all calls:  CacheInfo(hits=491, misses=14165, maxsize=None, currsize=14165)
    BB Best payoff  5.713
    BB Best final state  ((2, 1, 1, 1), (1, 1, 0, 1), (0, 0, 0, 1))
    BB action list  [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3), (2, 3), (0, 0)]
    BB Computation took 3.002690076828003 seconds


"""
import time
import numpy as np

# from mining import Mine, search_dp_dig_plan, search_bb_dig_plan

from mining import Mine, search_dp_dig_plan, search_bb_dig_plan

# from SOLUTION_mining import search_dp_rec


np.set_printoptions(3)

some_2d_underground_1 = np.array([
    [-0.814, 0.637, 1.824, -0.563],
    [0.559, -0.234, -0.366, 0.07],
    [0.175, -0.284, 0.026, -0.316],
    [0.212, 0.088, 0.304, 0.604],
    [-1.231, 1.558, -0.467, -0.371]])

some_3d_underground_1 = np.array([[[0.455, 0.579, -0.54, -0.995, -0.771],
                                   [0.049, 1.311, -0.061, 0.185, -1.959],
                                   [2.38, -1.404, 1.518, -0.856, 0.658],
                                   [0.515, -0.236, -0.466, -1.241, -0.354]],
                                  [[0.801, 0.072, -2.183, 0.858, -1.504],
                                   [-0.09, -1.191, -1.083, 0.78, -0.763],
                                   [-1.815, -0.839, 0.457, -1.029, 0.915],
                                   [0.708, -0.227, 0.874, 1.563, -2.284]],
                                  [[-0.857, 0.309, -1.623, 0.364, 0.097],
                                   [-0.876, 1.188, -0.16, 0.888, -0.546],
                                   [-1.936, -3.055, -0.535, -1.561, -1.992],
                                   [0.316, 0.97, 1.097, 0.234, -0.296]]])


def test_2D_search_dig_plan():
    # np.random.seed(10)

    # x_len, z_len = 5,4
    # some_neg_bias = -0.2
    # my_underground = np.random.randn(x_len, z_len) + some_neg_bias

    my_underground = some_2d_underground_1
    mine = Mine(my_underground)

    # #mine.initial += 0
    # # mine.initial = np.array([4, 3, 2, 3, 3])
    # mine.console_display()
    # print("Initial:")
    # a = mine.initial
    # print(a)
    # # print(mine.surface_neigbhours((2,)))
    # print("Cumsum:")
    # print(mine.cumsum_mine)
    # print("is_dangerous:")
    # print(mine.is_dangerous(mine.initial))
    # print("Payoff:")
    # print(mine.payoff(mine.initial))
    # print("Actions:")
    # g = mine.initial
    # for a in mine.actions(g):
    #     print(a)

    # print(my_underground.__repr__())

    print('-------------- DP computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state = search_dp_dig_plan(mine)
    toc = time.time()
    print('DP Best payoff ',best_payoff)
    print('DP Best final state ', best_final_state)
    print('DP action list ', best_a_list)
    print('DP Computation took {} seconds\n'.format(toc-tic))

    print('-------------- BB computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state = search_bb_dig_plan(mine)
    toc = time.time()
    print('BB Best payoff ',best_payoff)
    print('BB Best final state ', best_final_state)
    print('BB action list ', best_a_list)
    print('BB Computation took {} seconds'.format(toc-tic))


def test_3D_search_dig_plan():
    # np.random.seed(10)

    # x_len,y_len,z_len = 3,4,5
    # some_neg_bias = -0.3
    # my_underground = np.random.randn(x_len,y_len,z_len) + some_neg_bias

    my_underground = some_3d_underground_1
    mine = Mine(my_underground)

    # COMMENTED OUT TEMPORARILY:
    # # mine.initial += 1
    # # mine.initial = np.array([4, 3, 2, 3, 3])
    # mine.console_display()
    # print("Initial:")
    # print(mine.initial)
    # # print(mine.surface_neigbhours((2,)))
    # print("Cumsum:")
    # print(mine.cumsum_mine)
    # print("is_dangerous:")
    # print(mine.is_dangerous(mine.initial))
    # print("Payoff:")
    # print(mine.payoff(mine.initial))
    # print("Actions:")
    # for a in mine.actions(mine.initial):
    #     print(a)
    # print('yo yo')



    print('-------------- DP computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state = search_dp_dig_plan(mine)
    toc = time.time()
    print('DP Best payoff ', best_payoff)
    print('DP Best final state ', best_final_state)
    print('DP action list ', best_a_list)
    print('DP Computation took {} seconds\n'.format(toc - tic))

    print('-------------- BB computations -------------- ')
    tic = time.time()
    best_payoff, best_a_list, best_final_state = search_bb_dig_plan(mine)
    toc = time.time()
    print('BB Best payoff ', best_payoff)
    print('BB Best final state ', best_final_state)
    print('BB action list ', best_a_list)
    print('BB Computation took {} seconds'.format(toc - tic))



if __name__ == '__main__':
    pass
    print('= ' * 20)
    test_2D_search_dig_plan()
    print('= ' * 20)
    test_3D_search_dig_plan()