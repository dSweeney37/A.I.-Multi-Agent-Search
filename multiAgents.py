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
import random, util, pacman

from game import Agent





"""
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
"""
class ReflexAgent(Agent):
    def getAction(self, gameState):
        # Collect legal moves and successor states
        moves = gameState.getLegalActions()
        # Choose one of the best moves
        scores = [self.evaluationFunction(gameState, move) for move in moves]


        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best.
        chosenIndex = random.choice(bestIndices)

        return moves[chosenIndex]



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
    def evaluationFunction(self, currentGameState, move):
        # Generates Pacman's successor game state.
        gameState = currentGameState.generatePacmanSuccessor(move)
        # Gets Pacman's coordinates in the new game state.
        pos = gameState.getPacmanPosition()
        # Splits Pacman's coordinates into seperate variables.
        (posX, posY) = pos
        # A list containing the coordinates of the food pellets.
        foodList = gameState.getFood().asList()
        # A list for storing the distance to each food pellet.
        foodDistances = []


        # Adds the distance to each food pellet to the foodDistances list.
        for food in foodList:
            # Splits the food's coordinates into seperate variables.
            (foodX, foodY) = food
            # Calculates the number of moves from Pacman to the food pellet.
            distance = abs(posX - foodX) + abs(posY - foodY)


            foodDistances.append(distance)


        # If foodDistances is not empty, then return the game state score + the reciprocal of the
        # minimum value in the foodDistances list, otherwise only the game state score is returned.
        if foodDistances: return gameState.getScore() + (1 / min(foodDistances))
        else: return gameState.getScore()




"""
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
"""
def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()




"""
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *DO NOT* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated. It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
"""
class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)



    # Determines whether the current state is a terminal state.
    def isTerminal(self, gameState):
        return gameState.isLose() or gameState.isWin()



    # Determines if the max depth has been reached.
    def isMaxDepth(self, depth):
        return depth == self.depth



    # Takes a game state and agent index and returns the next agent's index.
    def getNextAgentIndex(self, gameState, agentIndex):
        # Retrieves the total number of agents in the game.
        numOfAgents = gameState.getNumAgents()
        # Determines the index of the next agent.
        nextAgentIndex = (agentIndex + 1) % numOfAgents


        return nextAgentIndex




class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        # Pacman's index.
        agentIndex = 0
        # The initial state's depth.
        depth = 0
        # A list of all of the actions that Pacman can make.
        moves = gameState.getLegalActions(agentIndex)
        # The next agent's index.
        nextAgentIndex = self.getNextAgentIndex(gameState, agentIndex)


        # Retrieves a list of minimax scores for the successor states.
        scores = [self.minValue(gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth) for move in moves]

        # Returns the action that leads to the highest minimax value.
        return moves[scores.index(max(scores))]



    # Determines the best action to make for a min agent.
    def minValue(self, gameState, agentIndex, depth):
        # A list of all of the actions that the min player can make.
        moves = gameState.getLegalActions(agentIndex)
        # The next agent's index.
        nextAgentIndex = self.getNextAgentIndex(gameState, agentIndex)


        # Returns the score of the game state if the state is a terminal state.
        if self.isTerminal(gameState): return self.evaluationFunction(gameState)

        # Retrieves a list of minimax scores for successor states.
        # If the next agent is a max agent then the depth coutner is incremented
        # and maxValue is called, otherwise minValue is called.
        if nextAgentIndex == 0: scores = [self.maxValue(gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth + 1) for move in moves]
        else: scores = [self.minValue(gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth) for move in moves]

        # Returns the minimum score value.
        return min(scores)



    # Determines the best action to make for a max agent.
    def maxValue(self, gameState, agentIndex, depth):
        # A list of all of the actions that the max player can make.
        moves = gameState.getLegalActions(agentIndex)
        # The next agent's index.
        nextAgentIndex = self.getNextAgentIndex(gameState, agentIndex)


        # Returns the score of the game state if the state is a terminal state or the limited depth has been reached.
        if self.isTerminal(gameState) or self.isMaxDepth(depth): return self.evaluationFunction(gameState)

        # Retrieves a list of minimax scores for successor states.
        scores = [self.minValue(gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth) for move in moves]

        # Returns the maximum score value.
        return max(scores)




class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        # Pacman's index.
        agentIndex = 0
        # Alpha's starting value.
        alpha = float('-inf')
        # Beta's starting value.
        beta = float('inf')
        # The initial state's depth.
        depth = 0
        # A list of all of the actions that Pacman can make.
        moves = gameState.getLegalActions(agentIndex)
        # The next agent's index.
        nextAgentIndex = self.getNextAgentIndex(gameState, agentIndex)
        # A list for keeping track of the root node's successor's scores.
        scores = []


        # Retrieves a list of alpha/beta scores for successor states.
        for move in moves:
            score = self.minValue(gameState.generateSuccessor(0, move), nextAgentIndex, depth, alpha, beta)


            # Appends the successor's score to the scores list.
            scores.append(score)
            # Set alpha to the maximum value of alpha and score.
            alpha = max(alpha, score)

        # Returns the action that leads to the highest alpha/beta value.
        return moves[scores.index(max(scores))]



    # Determines the best action to make for a min agent.
    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        # A list of all of the actions that the min player can make.
        moves = gameState.getLegalActions(agentIndex)
        # The next agent's index.
        nextAgentIndex = self.getNextAgentIndex(gameState, agentIndex)
        # Sets the initial value of score for minValue processing.
        score = float('inf')


        # Returns the score of the game state if the state is a terminal state.
        if self.isTerminal(gameState): return self.evaluationFunction(gameState)

        # Retrieves a list of alpha/beta scores for successor states.
        # If the next agent is a max agent then the depth counter is incremented
        # and maxValue is called, otherwise minValue is called.
        for move in moves:
            if nextAgentIndex == 0: score = min(score, self.maxValue(gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth + 1, alpha, beta))
            else: score = min(score, self.minValue(gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth, alpha, beta))


            # If the value of score < alpha, prune the remainder of the successors.
            if score < alpha: return score
            # Set beta to the minimum value of beta and score.
            beta = min(beta, score)

        # Returns score's value.
        return score



    # Determines the best action to make for a max agent.
    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        # A list of all of the actions that the max player can make.
        moves = gameState.getLegalActions(agentIndex)
        # The next agent's index.
        nextAgentIndex = self.getNextAgentIndex(gameState, agentIndex)
        # Sets the initial value of score for maxValue processing.
        score = float('-inf')


        # Returns the score of the game state if the state is a terminal state or the limited depth has been reached.
        if self.isTerminal(gameState) or self.isMaxDepth(depth): return self.evaluationFunction(gameState)

        # Retrieves a list of alpha/beta scores for successor states.
        for move in moves:
            score = max(score, self.minValue(gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth, alpha, beta))


            # If the value of score > beta, prune the remainder of the successors.
            if score > beta: return score
            # Set alpha to the maximum value of alpha and score.
            alpha = max(alpha, score)

        # Returns score's value.
        return score




class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        # Pacman's index.
        agentIndex = 0
        # The initial state's depth.
        depth = 0
        # A list of all of the actions that Pacman can make.
        moves = gameState.getLegalActions(agentIndex)
        # The next agent's index.
        nextAgentIndex = self.getNextAgentIndex(gameState, agentIndex)


        # Retrieves a list of expectimax scores for successor states.
        scores = [self.expectedValue(gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth) for move in moves]

        # Returns the action that leads to the highest minimax value.
        return moves[scores.index(max(scores))]



    def expectedValue(self, gameState, agentIndex, depth):
        # A list of all of the actions that the adversarial player can make.
        moves = gameState.getLegalActions(agentIndex)
        # The next agent's index.
        nextAgentIndex = self.getNextAgentIndex(gameState, agentIndex)
        # Initializes probability to 0 to account for cases where the agent has no legal actions.
        probability = 0


        # The uniform value for taking an action (e.g. 1 / # of legal actions).
        if moves: probability = 1 / len(moves)

        # Returns the score of the game state if the state is a terminal state.
        if self.isTerminal(gameState): return self.evaluationFunction(gameState)

        # Retrieves a list of expectimax scores for successor states.
        # If the next agent is a max agent then the depth coutner is incremented
        # and maxValue is called, otherwise expectedValue is called.
        if nextAgentIndex == 0: scores = [self.maxValue(gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth + 1) for move in moves]
        else: scores = [self.expectedValue(gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth) for move in moves]

        # Returns the sum of the successor states * the probability of each state.
        return sum(scores) * probability



    # Determines the best action to make for a max agent.
    def maxValue(self, gameState, agentIndex, depth):
        # A list of all of the actions that the min player can make.
        moves = gameState.getLegalActions(agentIndex)
        # The next agent's index.
        nextAgentIndex = self.getNextAgentIndex(gameState, agentIndex)


        # Returns the score of the game state if the state is a terminal state or the limited depth has been reached.
        if self.isTerminal(gameState) or self.isMaxDepth(depth): return self.evaluationFunction(gameState)

        # Retrieves a list of expectimax scores for successor states.
        scores = [self.expectedValue(gameState.generateSuccessor(agentIndex, move), nextAgentIndex, depth) for move in moves]

        # Returns the maximum score value.
        return max(scores)

