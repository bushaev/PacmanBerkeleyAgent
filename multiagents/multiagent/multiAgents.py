# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodList = newFood.asList()
        foodCount = newFood.count()
        foodDistance = [util.manhattanDistance(newPos, food) for food in foodList]

        minFood = 0
        if len(foodDistance) > 0:
            minFood = min(foodDistance)

        dist = min(
            [util.manhattanDistance(newPos, newGhostStates[i].getPosition()) for i in range(len(newGhostStates))])

        if dist < 3:
            if sum(newScaredTimes) > 0:
                dist = 1000
            else:
                dist = -1000
        else:
            dist = 0

        w = [3,
             -1,
             -1,
             1,
             1
             ]
        h = [
            successorGameState.getScore() - currentGameState.getScore(),
            minFood,
            foodCount,
            sum(newScaredTimes),
            dist,
        ]

        return sum([h[i] * w[i] for i in range(len(w))])


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in e Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def miniMax(self, gameState):
        return self.max_value(gameState, 0, -1)[1]

    def max_value(self, gameState, agentIndex, depth):
        depth += 1
        if depth == self.depth:
            return self.evaluationFunction(gameState), None

        actions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]

        values = [self.min_value(suc, agentIndex + 1, depth)[0] for suc in successors]
        if len(values) == 0:
            return self.evaluationFunction(gameState), None
        index = values.index(max(values))

        return values[index], actions[index]

    def min_value(self, gameState, agentIndex, depth):

        actions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]

        fun = self.min_value if agentIndex < self.agents - 1 else self.max_value
        values = [fun(suc, (agentIndex + 1) % self.agents, depth)[0] for suc in successors]
        if len(values) == 0:
            return self.evaluationFunction(gameState), None
        index = values.index(min(values))

        return values[index], actions[index]

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        self.agents = gameState.getNumAgents()
        return self.miniMax(gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def miniMax(self, gameState):

        return self.max_value(gameState, 0, -1, -99999999, 99999999)[1]

    def max_value(self, gameState, agentIndex, depth, alpha, beta):
        depth += 1
        if depth == self.depth:
            return self.evaluationFunction(gameState), None

        actions = gameState.getLegalActions(agentIndex)
        bestValue = -99999999
        bestAction = None
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            value = self.min_value(successor, agentIndex + 1, depth, alpha, beta)[0]

            if value > beta:
                return value, None  # pruning, no need to return action
            if bestValue < value:
                bestValue = value
                bestAction = action

            alpha = max(alpha, value)

        if bestAction is None:
            bestValue = self.evaluationFunction(gameState)

        return bestValue, bestAction

    def min_value(self, gameState, agentIndex, depth, alpha, beta):

        fun = self.min_value if agentIndex < self.agents - 1 else self.max_value
        actions = gameState.getLegalActions(agentIndex)
        bestValue = 99999999
        bestAction = None
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            value = fun(successor, (agentIndex + 1) % self.agents, depth, alpha, beta)[0]

            if value < alpha:
                return value, None  # pruning, no need to return action
            if bestValue > value:
                bestValue = value
                bestAction = action

            beta = min(beta, value)

        if bestAction is None:
            bestValue = self.evaluationFunction(gameState)

        return bestValue, bestAction

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.agents = gameState.getNumAgents()
        return self.miniMax(gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def max_value(self, gameState, agentIndex, depth):
        depth += 1
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        actions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        values = [self.prob_value(suc, agentIndex + 1, depth)[0] for suc in successors]
        if len(values) == 0:
            return self.evaluationFunction(gameState), None
        m = max(values)
        index = random.choice([i for i, j in enumerate(values) if j == m])
        return values[index], actions[index]

    def prob_value(self, gameState, agentIndex, depth):
        actions = gameState.getLegalActions(agentIndex)
        if gameState.isWin():
            return self.evaluationFunction(gameState), None

        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        fun = self.prob_value if agentIndex < self.agents - 1 else self.max_value
        values = [fun(suc, (agentIndex + 1) % self.agents, depth)[0] for suc in successors]
        actionLength = len(actions)
        if len(values) == 0:
            self.evaluationFunction(gameState), None
        v = [value * 1.0 / actionLength for value in values]
        return sum(v), None

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        self.agents = gameState.getNumAgents()
        r = self.max_value(gameState, 0, -1)
        return r[1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    from searchAgents import mazeDistance
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodList = newFood.asList()
    foodCount = newFood.count()
    foodDistance = [manhattanDistance(newPos, food) for food in foodList]

    minFood = 0
    if len(foodDistance) > 0:
        minFood = min(foodDistance)

    dist = min(
        [util.manhattanDistance(newPos, newGhostStates[i].getPosition()) for i in range(len(newGhostStates))])

    if dist < 2:
        if sum(newScaredTimes) > 0:
            dist = 1000
        else:
            dist = -1000
    else:
        dist = 0

    score = currentGameState.getScore()
    w = [
        5,
        -2,
        -2,
        1,
        0.5,
    ]
    h = [
        score,
        minFood,
        foodCount,
        sum(newScaredTimes),
        dist,
    ]

    return sum([h[i] * w[i] for i in range(len(w))])


# Abbreviation
better = betterEvaluationFunction
