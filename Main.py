#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np

from mining import Mine, search_dp_dig_plan, search_bb_dig_plan

def test_2D_search_dig_plan():

    np.random.seed(10)

    x_len, z_len = 5,4
    some_neg_bias = -0.2
    my_underground = np.random.randn(x_len, z_len) + some_neg_bias

    mine = Mine(my_underground)

    print('-------------- 2d mine -------------- ')
    Mine.console_display(mine)

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
    np.random.seed(10)

    x_len,y_len,z_len = 3,4,5
    some_neg_bias = -0.3
    my_underground = np.random.randn(x_len,y_len,z_len) + some_neg_bias

    mine = Mine(my_underground)

    print('-------------- 3d mine -------------- ')
    Mine.console_display(mine)

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