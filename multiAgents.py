# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        minFoodDist, minGhostDist = 0, 0
        if len(foodList) > 0:
            minFoodDist, closestFood = min((manhattanDistance(newPos, food), food) for food in foodList)
        if len(newGhostStates) > 0:
            minGhostDist, closestGhost = min((manhattanDistance(newPos, ghost.getPosition()), ghost) for ghost in newGhostStates)
        ghostScore = 0
        if minGhostDist < 3 and closestGhost.scaredTimer <= 0:
            ghostScore = -100
        # foodScore = minFoodDist + len(foodList), add 1 in case zero sum
        score = 1.0 / (minFoodDist + len(foodList) + 1) + ghostScore
        return score + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

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
        "*** YOUR CODE HERE ***"
        depth = self.depth
        actions = gameState.getLegalActions(0)
        random.shuffle(actions)
        if len(actions) > 0:
            # max(pacman), for each action we calculate the maximum score
            maximum, bestAction = max((self.miniMax(gameState.generateSuccessor(0, action), depth, 1), action) for action in actions)
            return bestAction
        else:
            return None

    def miniMax(self, gameState, depth, index):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if index == 0:
            # means the current agent is Pacman, do Max
            actions = gameState.getLegalActions(0)
            if len(actions) > 0:
                return max(self.miniMax(gameState.generateSuccessor(0, action), depth, 1) for action in actions)
            else:
                return self.evaluationFunction(gameState)
        else:
            actions = gameState.getLegalActions(index)
            nextIndex = (index + 1) % gameState.getNumAgents()
            if nextIndex == 0:
                # nextAgent is Pacman, we decrease the depth
                depth -= 1
            if len(actions) > 0:
                return min(self.miniMax(gameState.generateSuccessor(index, action), depth, nextIndex) for action in actions)
            else:
                return self.evaluationFunction(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        alpha = float('-inf')
        beta = float('inf')
        bestAction = None
        maxVal = float('-inf')
        for action in gameState.getLegalActions(0):
            val = self.alphaBetaPrun(gameState.generateSuccessor(0, action), depth, alpha, beta, 1)
            if val > maxVal:
                maxVal = val
                bestAction = action
            if maxVal > beta:
                return bestAction
            alpha = max(alpha, maxVal)
        return bestAction

    def alphaBetaPrun(self, gameState, depth, alpha, beta, index):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        elif index == 0:
            # means the current agent is Pacman, do Max
            maxVal = float('-inf')
            actions = gameState.getLegalActions(0)
            for action in actions:
                val = self.alphaBetaPrun(gameState.generateSuccessor(0, action), depth, alpha, beta, 1)
                if val > maxVal:
                    if depth == self.depth:
                        bestAction = action
                    maxVal = val
                if maxVal > beta:
                    return maxVal
                alpha = max(alpha, maxVal)
            return maxVal
        else:
            nextIndex = (index + 1) % gameState.getNumAgents()
            if nextIndex == 0:
                # nextAgent is Pacman, we decrease the depth
                depth -= 1
            minVal = float('inf')
            actions = gameState.getLegalActions(index)
            for action in actions:
                minVal = min(minVal, self.alphaBetaPrun(gameState.generateSuccessor(index, action), depth, alpha, beta, nextIndex))
                if minVal < alpha:
                    return minVal
                beta = min(beta, minVal)
            return minVal

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        maximum = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            v = self.expectiMax(gameState.generateSuccessor(0, action), self.depth, 1)
            if v > maximum:
                maximum = v
                bestAction = action
        return bestAction

    def expectiMax(self, gameState, depth, index):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if index == 0:
            actions = gameState.getLegalActions(0)
            return max(self.expectiMax(gameState.generateSuccessor(0, action), depth, 1) for action in actions)
        else:
            actions = gameState.getLegalActions(index)
            nextIndex = (index + 1) % gameState.getNumAgents()
            if nextIndex == 0:
                depth -= 1
            sumVal = sum(self.expectiMax(gameState.generateSuccessor(index, action), depth, nextIndex) for action in actions)
            return sumVal * 1.0 / len(actions) * 1.0


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <I consider ghost's distace to pacman is important, the closer
      the ghost to pacman, the worse score it should return.
      Also, if less left foods left, higher score. >
    """
    "*** YOUR CODE HERE ***"
    foodList = currentGameState.getFood().asList()
    currentGhoastStates = currentGameState.getGhostStates()
    currentPos = currentGameState.getPacmanPosition()
    minGhostDist = 0
    for ghost in currentGhoastStates:
        if ghost.scaredTimer > 0 and manhattanDistance(ghost.getPosition(), currentPos) > 3:
            currentGhoastStates.remove(ghost)
    if len(currentGhoastStates) > 0:
        minGhostDist = min(manhattanDistance(ghost.getPosition(), currentPos) for ghost in currentGhoastStates)
    minDistToFood = min(0, (manhattanDistance(currentPos, food) for food in foodList))
    return currentGameState.getScore() - minGhostDist + 100/(len(foodList) + minDistToFood + 1)


# Abbreviation
better = betterEvaluationFunction

