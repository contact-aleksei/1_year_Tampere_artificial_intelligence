# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import time

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    from game import Directions
    visited = set() # unique elements
    state = problem.getStartState()
    #returns starting agent's position
    waiting_list = util.Stack()
    # LIFO
    # last in first out
    # parents = collections.defaultdict(collections.UserDict)
    parents = {}
    #dictionary
    sequence = []
    #LIFO
    for action in problem.getSuccessors(state):
        # in order to push full-state values
        waiting_list.push(action)
        # enumarating tuple

    while not waiting_list.isEmpty():
        state = waiting_list.pop()
        
        visited.add(state[0])
        # node is visited and we wont visit those nodes
        
        for substate in problem.getSuccessors(state[0]):
        # take a look to successors of current node
        
            if substate[0] not in visited:
                # if not in visited 
                # saving parents
                parents[substate[0]]={'parent':state}                                               
                # generate new node
                waiting_list.push(substate)
                # push to stack
                if problem.isGoalState(substate[0]): 
                    target_state = substate                                                                                                  
                    #finding wayback


    while target_state[0] in parents.keys():
        temp=parents[target_state[0]]['parent']
        sequence.append(target_state[1])
        target_state = temp
    sequence.append(target_state[1])
    return sequence[::-1]   

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    waiting_list = util.Queue()
    # QUEUE
    # FIFO 
    visited = set()
    parents = {}
    #collections.defaultdict(collections.UserDict)
    sequence = []
    start_state = problem.getStartState()
    for action in problem.getSuccessors(start_state):
        # in order to push full-state values
        waiting_list.push(action)
        
    while not waiting_list.isEmpty():
        node = waiting_list.pop()
        visited.add(node[0])
        for action in problem.getSuccessors(node[0]):
            
            #if child.STATE is not in explored or frontier then
            if action[0] not in visited:
                parents[action[0]] = {'parent':node}                                               
                waiting_list.push(action)
                if problem.isGoalState(action[0]):
                    target_state = action           
                        
                                                                               
    while target_state[0] in parents.keys():
        temp=parents[target_state[0]]['parent']
        sequence.append(target_state[1])
        target_state = temp
    sequence.append(target_state[1])
    return sequence[::-1]

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
# An element with high priority is dequeued before an element with low priority.
# If two elements have the same priority, they are served according to their order in the queue.    
    pq_state = util.PriorityQueue()
# Returns the start state for the search problem
    start = problem.getStartState()
#                 node, actions, visited
    pq_state.push((start, [], list()),0)
    
    while not pq_state.isEmpty():
        
        # check if node is visited
        node, actions, visited = pq_state.pop()
        if node in visited: 
            continue
        
        # check if node is Goal State, returns True if it is goal state
        if problem.isGoalState(node):
            return actions
        
        # otherwise we append node
        visited.append(node)
        
        # calculating actions and costs
        for coord, direction, _ in problem.getSuccessors(node):
            n_actions = actions + [direction]
            
            # we use FUNCTION getCostOfActions
            pq_state.push((coord, n_actions, visited), problem.getCostOfActions(n_actions))
                                                            # cost of list of actions
            
     
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from game import Actions

    waiting_list = util.PriorityQueue()
    COSTS = {}
    start_state = problem.getStartState()
    COSTS[start_state] = 0
    waiting_list.push(start_state,0)
    parents = {}
    
    while not waiting_list.isEmpty():
        q_state = waiting_list.pop()
        if problem.isGoalState(q_state):
            target_state = q_state
            break
        for child in problem.getSuccessors(q_state):
            n_cost = COSTS[q_state] + child[2]
            
            if child[0] not in COSTS or n_cost < COSTS[q_state]:
                COSTS[child[0]] = n_cost
                prior = n_cost + heuristic(child[0], problem)
                waiting_list.push(child[0], prior)
                parents[child[0]] = q_state

    sequence = []
    prev_state = target_state
    while target_state in parents.keys():
        target_state = parents[target_state]
        direction = Actions.vectorToDirection([prev_state[0] - target_state[0], prev_state[1] - target_state[1]])
        prev_state = target_state
        sequence.append(direction)
        
    return sequence[::-1]


# Abbreviatons
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
