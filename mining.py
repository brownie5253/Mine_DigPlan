#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:56:47 2021

@author: frederic

# =============================================================================
#                                  CHANGELOG:
#   - Fixed Mine class, input arrays will be formatted as (x, z) for 2D and 
#     (x, y, z) for 3D. - Ethan
#   - States are now correctly working, e.g. doing state(action) will now find
#     the correct location in the mine. - Ethan
#   - Added variable self.three_dim, set to True if mine is 3D - Ethan
#   - Implemented Is_Dangerous function, seems to be working correctly - Ethan
#   - changed code in init from 0 to -2 for x_len as in 3d mind x is the second index and -2 for 2d mine will index first variable - Connor
#   - THIS A COMMENT BUT I'VE EDITED IT
#  - Matti - for git
# 
#   - THIS IS ANOTHER COMMENT
#   - I am confused
#   Big mood
# 
# =============================================================================

class problem with     

An open-pit mine is a grid represented with a 2D or 3D numpy array. 

The first coordinates are surface locations.

In the 2D case, the coordinates are (x, z).
In the 3D case, the coordinates are (x, y, z).
The last coordinate 'z' points down.

    
A state indicates for each surface location  how many cells 
have been dug in this pit column.

For a 3D mine, a surface location is represented with a tuple (x, y).

For a 2D mine, a surface location is represented with a tuple (x, ).


Two surface cells are neighbours if they share a common border point.
That is, for a 3D mine, a surface cell has 8 surface neighbours.


An action is represented by the surface location where the dig takes place.


"""
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import functools

from numbers import Number

import search

def my_team():

     '''    Return the list of the team members of this assignment submission
     as a list of triplet of the form (student_number, first_name, last_name)        '''

     return [ (10467858, 'Ethan', 'Griffiths'), (10467874, 'Mattias', 'Winsen'), (10486925, 'Connor', 'Browne') ]

def convert_to_tuple(a):
    '''
    Convert the parameter 'a' into a nested tuple of the same shape as 'a'.
    
    The parameter 'a' must be array-like. That is, its elements are indexed.

    Parameters
    ----------
    a : flat array or an array of arrays

    Returns
    -------
    the conversion of 'a' into a tuple or a tuple of tuples

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return ()
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return tuple(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return tuple(tuple(r) for r in a)


def convert_to_list(a):
    '''
    Convert the array-like parameter 'a' into a nested list of the same 
    shape as 'a'.

    Parameters
    ----------
    a : flat array or array of arrays

    Returns
    -------
    the conversion of 'a' into a list or a list of lists

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return []
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return list(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return [list(r) for r in a]




class Mine(search.Problem):
    '''
    
    Mine represent an open mine problem defined by a grid of cells 
    of various values. The grid is called 'underground'. It can be
    a 2D or 3D array.
    
    The z direction is pointing down, the x and y directions are surface
    directions.
    
    An instance of a Mine is characterized by 
    - self.underground : the ndarray that contains the values of the grid cells
    - self.dig_tolerance : the maximum depth difference allowed between 
                           adjacent columns 
    
    Other attributes:
        self.len_x, self.len_y, self.len_z : int : underground.shape
        self.cumsum_mine : float array : cumulative sums of the columns of the 
                                         mine
    
    A state has the same dimension as the surface of the mine.
    If the mine is 2D, the state is 1D.
    If the mine is 3D, the state is 2D.
    
    state[loc] is zero if digging has not started at location loc.
    More generally, state[loc] is the z-index of the first cell that has
    not been dug in column loc. This number is also the number of cells that
    have been dugged in the column.
    
    States must be tuple-based.
    
    '''

    def __init__(self, underground, dig_tolerance = 1):
        '''
        Constructor
        
        Initialize the attributes
        self.underground, self.dig_tolerance, self.len_x, self.len_y, self.len_z, 
        self.cumsum_mine, and self.initial
        
        The state self.initial is a filled with zeros.

        Parameters
        ----------
        underground : np.array
            2D or 3D. Each element of the array contains 
            the profit value of the corresponding cell.
        dig_tolerance : int
             Mine attribute (see class header comment)
        Returns
        -------
        None.

        '''

        # self.underground should be considered as a 'read-only' variable!
        self.underground = underground

        assert dig_tolerance >= 1, "Dig tolerance bust be greater than 0 otherwise no actions possible"
        self.dig_tolerance = dig_tolerance
        assert underground.ndim in (2, 3)

        # Determine if mine is 3D or not
        if self.underground.ndim == 3:
            self.three_dim = True
        else:
            self.three_dim = False

        # 0 axis of underground is always x axis
        x_axis = 0
        # -1 axis of underground is always z axis
        z_axis = -1

        self.len_z = self.underground.shape[z_axis]
        self.len_x = self.underground.shape[x_axis]

        # 3D mine case
        if self.three_dim:
            self.len_y = self.underground.shape[1]
            self.initial = np.zeros((self.len_x, self.len_y), dtype=int)

        # 2D mine case            
        else:
            # 0 axis of underground
            self.len_y = 0
            self.initial = np.zeros(self.len_x, dtype=int)

        self.cumsum_mine = np.cumsum(self.underground, dtype=float, axis=-1)


    def surface_neigbhours(self, loc):
        '''
        Return the list of neighbours of loc

        Parameters
        ----------
        loc : surface coordinates of a cell.
            a singleton (x, ) in case of a 2D mine
            a pair (x, y) in case of a 3D mine

        Returns
        -------
        A list of tuples representing the surface coordinates of the
        neighbouring surface cells.

        '''
        L=[]
        assert len(loc) in (1, 2)
        if len(loc)==1:
            if loc[0]-1>=0:
                L.append((loc[0]-1, ))
            if loc[0]+1<self.len_x:
                L.append((loc[0]+1, ))
        else:
            # len(loc) == 2
            for dx, dy in ((-1, -1), (-1, 0), (-1, +1), 
                          (0, -1), (0, +1), 
                          (+1, -1), (+1, 0), (+1, +1)):
                if  (0 <= loc[0]+dx < self.len_x) and (0 <= loc[1]+dy < self.len_y):
                    L.append((loc[0]+dx, loc[1]+dy))
        return L


    def state_indexes(self):
        x_coordinates = np.arange(self.len_x)

        # 3D case
        if self.three_dim:
            y_coordinates = np.arange(self.len_y)
            args = (convert_to_list(x_coordinates), convert_to_list(y_coordinates))
            pools = [tuple(pool) for pool in args]
            coordinates = [[]]
            for pool in pools:
                coordinates = [x + [y] for x in coordinates for y in pool]
            #check if not enough coordinates
            assert len(coordinates) == self.len_x * self.len_y, "less coordinates and there should be, state_indexes"
        # 2D case
        else:
            coordinates = convert_to_list(x_coordinates)
            # check if not enough coordinates
            assert len(coordinates) == self.len_x, "less coordinates and there should be, state_indexes"

        return coordinates


    def not_bottom(self, state, action_loc):
        """Check if the state is at the bottom of the mine for the given action.
        Returns a bool containing the result of this test.

        Parameters
        ----------
        state :
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine
        action_loc :
             int value/s contained in a tuple. Represents and x or x,y position in state
        Returns
        -------
        Returns True if the current dug level is not at the bottom (z axis) of underground
        """

        return (state[action_loc] < self.len_z)


    def actions(self, state):
        '''
        Return a generator of valid actions in the given state 'state'
        An action is represented as a location. An action is deemed valid if
        it doesn't break the dig_tolerance constraint or is not at the bottom of the mine.

        Parameters
        ----------
        state : 
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine

        Returns
        -------
        a generator of valid actions

        '''
        state = np.array(state)

        state_indexs = self.state_indexes()

        for loc in state_indexs:
            action_loc = tuple([loc])
            if(self.three_dim):
                action_loc = tuple([loc[0], loc[1]])
            if (self.is_dangerous(self.result(state,action_loc)) == False and self.not_bottom(state, action_loc)):
                yield action_loc


    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must a valid actions.
        That is, one of those generated by  self.actions(state)."""
        action = tuple(action)
        new_state = np.array(state) # Make a copy
        new_state[action] += 1
        return convert_to_tuple(new_state)


    def console_display(self):
        '''
        Display the mine on the console

        Returns
        -------
        None.

        '''
        print('Mine of depth {}'.format(self.len_z))
        if self.underground.ndim == 2:
            # 2D mine
            print('Plane x, z view')
        else:
            # 3D mine
            print('Level by level x, y slices')
        #
        print(self.__str__())

    def __str__(self):
        if self.underground.ndim == 2:
            # 2D mine
            return str(self.underground.T)
        else:
            # 3D mine
            # level by level representation
            return '\n'.join('level {}\n'.format(z)
                   +str(self.underground[..., z]) for z in range(self.len_z))



            return self.underground[loc[0], loc[1], :]


    @staticmethod
    def plot_state(state):
        if state.ndim==1:
            fig, ax = plt.subplots()
            ax.bar(np.arange(state.shape[0]) , 
                    state
                    )
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            assert state.ndim==2
            # bar3d(x, y, z, dx, dy, dz, 
            # fake data
            _x = np.arange(state.shape[0])
            _y = np.arange(state.shape[1])
            _yy, _xx = np.meshgrid(_y, _x) # cols, rows
            x, y = _xx.ravel(), _yy.ravel()
            top = state.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1
            fig = plt.figure(figsize=(3, 3))
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title('State')
        #
        plt.show()

    def payoff(self, state):
        '''
        Compute and return the payoff for the given state.
        That is, the sum of the values of all the digged cells.

        Parameters
        ----------
        state :
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine

        Returns
        -------
        An int value corresponding to the payoff
        '''


        # convert to np.array in order to use tuple addressing
        # state[loc]   where loc is a tuple
        state = np.array(state)

        state_indexes = self.state_indexes()
        x_coordinates = np.arange(self.len_x)
        index_adjust = 1
        #Adjust state to make it an array of z indexes of last mined block for column
        z_coordinates = state - index_adjust

        # 3D case
        if self.three_dim:
            state_indexes = np.array(state_indexes)
            x_coordinates = state_indexes[:,0]
            y_coordinates = state_indexes[:,1]
            z_coordinates = np.concatenate(z_coordinates).ravel().tolist()
            cumsum_values = self.cumsum_mine[x_coordinates, y_coordinates, z_coordinates]
            # check if not enough values
            assert len(cumsum_values) == self.len_x * self.len_y, "less values and there should be, payoff"

        #2D case
        else:
            cumsum_values = self.cumsum_mine[x_coordinates, z_coordinates]
            # check if not enough values
            assert len(cumsum_values) == self.len_x, "less values and there should be, payoff"

        # if the dug amount in any state location was 0 it will now be -1
        check = (np.array(z_coordinates) >= 0)
        return np.sum((cumsum_values * check))


    def is_dangerous(self, state):
        '''
        determines if the current state breaches the dig_tolerance amount, making it dangerous

        Parameters
        ----------
        state :
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine

        Returns
        -------
        Return True iff the given state breaches the dig_tolerance constraints.

        '''
        # convert to np.array in order to use numpy operators
        state = np.array(state)

        # 3D case
        if self.three_dim:
            x_test = state[:, :-1] - state[:, 1:]
            y_test = state[:-1, :] - state[1:, :]
            diagonal_test_1 = state[:-1, :-1] - state[1:, 1:]
            diagonal_test_2 = np.rot90(state)[:-1, :-1] - np.rot90(state)[1:, 1:]

            # Concatenate all tests and check for unacceptable tolerances
            return(np.any(abs(np.concatenate((x_test, y_test, diagonal_test_1, diagonal_test_2),
                                             axis=None)) > self.dig_tolerance))
        # 2D case
        else:
            # Simply check along the x axis for unacceptable tolerances
            return(np.any(abs(state[:-1] - state[1:]) > self.dig_tolerance))

    # ========================  Class Mine  ==================================


def search_dp_dig_plan(mine):
    '''
    Search using Dynamic Programming the most profitable sequence of 
    digging actions from the initial state of the mine.
    
    Return the sequence of actions, the final state and the payoff
    

    Parameters
    ----------
    mine : a Mine instance

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''

    @functools.lru_cache(maxsize=None)
    def search_rec(state):
        #Inital state of return values
        best_payoff = mine.payoff(state)
        best_action_list = []
        best_final_state = state

        #Loop through each possible action of state as children states
        for child_action in mine.actions(state):
            child_state = mine.result(state, tuple(child_action))

            #Recursivly call search_rec for child state
            child_payoff, child_action_list, child_final_state = search_rec(child_state)

            #If the child of current state has better payoff, save it's returned values
            if (child_payoff > best_payoff):
                best_payoff = child_payoff
                best_action_list =  list([child_action]) + child_action_list
                best_final_state = child_final_state

        return best_payoff, best_action_list, best_final_state

    print("Before DP is run: ", search_rec.cache_info())
    best_payoff, best_action_list, best_final_state = search_rec(convert_to_tuple(mine.initial))
    print("After DP is run: ", search_rec.cache_info())

    return best_payoff, best_action_list, best_final_state

def search_bb_dig_plan(mine):
    '''
    Compute, using Branch and Bound, the most profitable sequence of 
    digging actions from the initial state of the mine.
        

    Parameters
    ----------
    mine : Mine
        An instance of a Mine problem.

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''

    @functools.lru_cache(maxsize=None) 
    def optimistic_payoff(state): 
        """Returns the best possible payoff for a node state, ignoring slope constraint."""
        ### Ideally need to redo this with array indexing if possible ###

        state = np.array(state) - 1 # subtract 1 to correctly index the cumsum array
        max_cumsum = np.empty(0) # empty array to hold best values in cumsum

        if mine.three_dim: # 3D Case
            for x,state_row in enumerate(state):
                for y,z in enumerate(state_row):
                    if z > -1:
                        max_cumsum = np.append(max_cumsum, np.amax(mine.cumsum_mine[x,y,z:]))
                    else: # case where there is the option to not dig at all (0 payoff is an option)
                        max_cumsum = np.append(max_cumsum, max(np.amax(mine.cumsum_mine[x,y]), 0))

        else: # 2D Case
            for x,z in enumerate(state):
                if z > -1:
                    max_cumsum = np.append(max_cumsum, np.amax(mine.cumsum_mine[x,z:]))
                else: # case where there is the option to not dig at all
                    max_cumsum = np.append(max_cumsum, max(np.amax(mine.cumsum_mine[x]), 0))
        
        return np.sum(max_cumsum)

    print(optimistic_payoff.cache_info()) # Cache Info 

    node = search.Node(convert_to_tuple(mine.initial)) 
    opt_pay = lambda x : optimistic_payoff(convert_to_tuple(x.state)) # f for Priority Queue will be the optimistic payoff
    # frontier = search.PriorityQueue('max',opt_pay) # Use prio queue to explore best optimistic nodes first
    frontier = search.FIFOQueue() # FIFO is faster, although likely due to optimistic_payoff(s) being slow
    frontier.append(node) # append first node

    # Initialise values for best node
    best_node = node
    best_payoff = mine.payoff(best_node.state)
    best_action_list = []
    best_final_state = best_node.state

    while frontier:
        node = frontier.pop()
        node_payoff = mine.payoff(node.state)

        # Check if node is obsolete by now (not needed for Prio queue, minimal gain anyways with cache enabled)
        # optimistic_node = opt_pay(node)
        # if optimistic_node <= best_payoff:
        #     continue
        
        # Store best node found
        if node_payoff >= best_payoff:
            best_node = node
            best_payoff = node_payoff

        # # # Debugging:
        # print('State:    ', node.state)
        # print('Best Pay: ', best_payoff)
        # print('Opt Pay:  ', opt_pay(node))
        # print('Payoff:   ', mine.payoff(node.state))
        # print()

        for child in node.expand(mine):
            # check that child has not been added to frontier and optimistic payoff is not worse than current payoff
            if child not in frontier and opt_pay(child) > best_payoff: 
                frontier.append(child)

    best_action_list = best_node.solution()
    best_final_state = convert_to_tuple(best_node.state)

    print(optimistic_payoff.cache_info()) # Cache Info

    return best_payoff, best_action_list, best_final_state

def find_action_sequence(s0, s1):
    '''
    Compute a sequence of actions to go from state s0 to state s1.
    There may be several possible sequences.
    
    Preconditions: 
        s0 and s1 are legal states, s0<=s1 and 
    
    Parameters
    ----------
    s0 : tuple based mine state
    s1 : tuple based mine state 

    Returns
    -------
    A sequence of actions to go from state s0 to state s1

    '''
    # safety check
    s0 = np.array(s0)
    s1 = np.array(s1)
    assert(s0.ndim == s1.ndim)

    # approach: among all columns for which s0 < s1, pick the column loc
    # with the smallest s0[loc]
    def find_sequence_3d(s0, s1, width):
        loc = 0
        output = []

        while True:
            for i in range(width): #loop through each "slice" of the 3d mine
                for location in range(len(s0[0])): 
                    if s0[i][location] == loc & s0[i][location] < s1[i][location]: #if current position is at the current level, and is less than the final position
                        output.append((i, location))
                        s0[i][location] += 1    
            loc += 1

            if np.all((s0 == s1)):
                return output
            
    if s0.ndim == 2: #if 3d (Mine.three_dim won't exist here)
        return tuple(find_sequence_3d(np.array(s0), np.array(s1), len(s0)))
    else: #if 2d
        return tuple(find_sequence_3d(np.array([s0]), np.array([s1]), 1))
