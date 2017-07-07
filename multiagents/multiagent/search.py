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

from util import Stack, Queue, PriorityQueue
from util import Counter

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


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
    return [s, s, w, s, w, w, s, w]


class State:
    def __init__(self, location, path):
        self.location = location
        self.path = path


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    # defining variables
    LOCATION = 0
    DIRECTION = 1

    stack = Stack()
    used = Counter()
    cur = State(problem.getStartState(), [])  # current (location, direction)
    stack.push(cur)

    while not stack.isEmpty():
        cur = stack.pop()  # take last item from stack
        used[cur.location] = 1  # mark as used

        if problem.isGoalState(cur.location):
            break

        for x in problem.getSuccessors(cur.location):
            if used[x[LOCATION]] is 0:
                stack.push(State(x[LOCATION], cur.path + [x[1]]))

    return cur.path


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    LOCATION = 0
    DIRECTION = 1

    q = Queue()
    used = set()
    cur = State(problem.getStartState(), [])  # current (location, path)
    q.push(cur)

    while not q.isEmpty():
        cur = q.pop()  # take last item from queue

        if cur.location in used:
            continue

        used.add(cur.location)  # mark as used

        if problem.isGoalState(cur.location):
            break

        for x in problem.getSuccessors(cur.location):
            if not x[LOCATION] in used:
                q.push(State(x[LOCATION], cur.path + [x[DIRECTION]]))

    return cur.path


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return aStarSearch(problem)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    pq = PriorityQueue()
    used = set()
    cur = State(problem.getStartState(), [])
    pq.push(cur, 0)
    while not pq.isEmpty():
        cur = pq.pop()

        if problem.isGoalState(cur.location):
            return cur.path

        if cur.location in used:
            continue

        used.add(cur.location)
        for x in problem.getSuccessors(cur.location):

            if x[0] not in used:
                pq.push(State(x[0], cur.path + [x[1]]),
                              problem.getCostOfActions(cur.path + [x[1]]) + heuristic(x[0], problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
