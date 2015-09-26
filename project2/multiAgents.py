# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import math

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


  def evaluationFunction(self,currentGameState, action):
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
    score = successorGameState.getScore()
    wallMap = successorGameState.getWalls()
    foodMap = currentGameState.getFood()
    height = foodMap.height
    width = foodMap.width
    foodPos= [(x,y) for x in range(width) for y in range(height) if foodMap[x][y]]
    distMap = [[None]*height for x in range(width)]
    # distMaps[(xFood, yFood)][x][y] should be the distance in the maze from (xFood, yFood) to (x, y)
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    # do BFS to fill the distMap
    bfsQueue = util.Queue()
    bfsQueue.push((newPos[0], newPos[1], 0))
    while not bfsQueue.isEmpty():
        x, y, dist = bfsQueue.pop()
        if distMap[x][y] is None:
            distMap[x][y] = dist
            for dx, dy in offsets:
                if not wallMap[x + dx][y + dy]:
                    bfsQueue.push((x + dx, y + dy, dist + 1))

    food_dists =[distMap[x_food][y_food] for x_food,y_food in foodPos]
    ghost_dists =[distMap[int(x_g)][int(y_g)] for x_g,y_g in successorGameState.getGhostPositions()]
    min_ghost_dist = min(ghost_dists)
    ghost_score=300/(min_ghost_dist+1) if min_ghost_dist < 3 else 0

    for i in range(len(newScaredTimes)):
      if newScaredTimes[i] is not 0:
        if ghost_dists[i] is min_ghost_dist:
          ghost_score = -1*ghost_score

    food_score = 30/(min(food_dists)+1)
    newScore = 0 +food_score - ghost_score +sum(newScaredTimes)
    return newScore



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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    score, action = self.evaluateGameStateMinimaxRecursive(gameState, self.depth, 0)
    #print "Debug: score for the best action (" + str(action) + ")is " + str(score)
    return action
  
  def evaluateGameStateMinimaxRecursive(self, gameState, remainingDepth, agent):
    # returns score, action
    # example of a sequence of recursive calls:
    # (depth 2, agent 0) => (depth 2, agent 1) => (depth 2, agent 2) => (d1, a0) => (d1, a1) => (d1, a2) => (d0, a0)
    if remainingDepth <= 0:
      return self.evaluationFunction(gameState), Directions.STOP
    if gameState.isLose() or gameState.isWin():
      return self.evaluationFunction(gameState), Directions.STOP
    
    numAgents = gameState.getNumAgents()
    isAdversary = (agent != 0)
    nextRemainingDepth = remainingDepth if agent + 1 < numAgents else remainingDepth - 1
    nextAgent = agent + 1 if agent + 1 < numAgents else 0
    
    print "getLegalActions " + str(remainingDepth) + "," + str(agent)
    actions = gameState.getLegalActions(agent)
    actions = [a for a in actions if a != Directions.STOP]
    actionScores = []
    if len(actions) == 0:
      return self.evaluationFunction(gameState), Directions.STOP
    for action in actions:
      nextGameState = gameState.generateSuccessor(agent, action)
      score, _ = self.evaluateGameStateMinimaxRecursive(nextGameState, nextRemainingDepth, nextAgent)
      actionScores.append(score)
    
    bestActionIndex = actionScores.index(min(actionScores) if isAdversary else max(actionScores))
    bestAction = actions[bestActionIndex]
    bestScore = actionScores[bestActionIndex]
    return bestScore, bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    score, action = self.evaluateGameStateMinimaxRecursive(gameState, self.depth, 0, None, None)
    #print "Debug: score for the best action (" + str(action) + ")is " + str(score)
    return action
  
  def evaluateGameStateMinimaxRecursive(self, gameState, remainingDepth, agent, alpha, beta):
    # returns score, action
    # example of a sequence of recursive calls:
    # (depth 2, agent 0) => (depth 2, agent 1) => (depth 2, agent 2) => (d1, a0) => (d1, a1) => (d1, a2) => (d0, a0)
    if remainingDepth <= 0:
      return self.evaluationFunction(gameState), Directions.STOP
    if gameState.isLose() or gameState.isWin():
      return self.evaluationFunction(gameState), Directions.STOP
    
    numAgents = gameState.getNumAgents()
    isAdversary = (agent != 0)
    nextRemainingDepth = remainingDepth if agent + 1 < numAgents else remainingDepth - 1
    nextAgent = agent + 1 if agent + 1 < numAgents else 0
    
    actions = gameState.getLegalActions(agent)
    actions = [a for a in actions if a != Directions.STOP]
    actionScores = []
    if len(actions) == 0:
      return self.evaluationFunction(gameState), Directions.STOP
    
    if isAdversary:
      bestScore = beta # inf
    else:
      bestScore = alpha # -inf
    bestAction = None
    nextAlpha = alpha
    nextBeta = beta
    
    for action in actions:
      nextGameState = gameState.generateSuccessor(agent, action)
      score, _ = self.evaluateGameStateMinimaxRecursive(nextGameState, nextRemainingDepth, nextAgent, nextAlpha, nextBeta)
      if isAdversary: # min node
        if bestScore is None or score < bestScore: # score is also smaller than nextBeta
          bestScore = nextBeta = score
          bestAction = action
        if alpha is not None and bestScore <= alpha:
          return alpha, bestAction
      else: # max node
        if bestScore is None or score > bestScore: # score is also bigger than nextAlpha
          bestScore = nextAlpha = score
          bestAction = action
        if beta is not None and bestScore >= beta:
          return beta, bestAction
    
    return bestScore, bestAction

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
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

