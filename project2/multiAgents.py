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
import random

from game import Agent
from game import Actions

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
    
    #print "getLegalActions " + str(remainingDepth) + "," + str(agent)
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
    score, action = self.evaluateGameStateExpectimaxRecursive(gameState, self.depth, 0)
    return action
    
  def evaluateGameStateExpectimaxRecursive(self, gameState, remainingDepth,agent):
    if remainingDepth <= 0:
      return self.evaluationFunction(gameState), Directions.STOP
    if gameState.isLose() or gameState.isWin():
      return self.evaluationFunction(gameState), Directions.STOP
    
    numAgents = gameState.getNumAgents()
    isExpect = (agent != 0)
    nextRemainingDepth = remainingDepth if agent + 1 < numAgents else remainingDepth - 1
    nextAgent = agent + 1 if agent + 1 < numAgents else 0
    
    actions = gameState.getLegalActions(agent)
    actions = [a for a in actions if a != Directions.STOP]
    actionScores = []
    if len(actions) == 0:
      return self.evaluationFunction(gameState), Directions.STOP
    for action in actions:
      nextGameState = gameState.generateSuccessor(agent, action)
      score, nextAction = self.evaluateGameStateExpectimaxRecursive(nextGameState, nextRemainingDepth, nextAgent)
      actionScores.append(score)

    if isExpect:
      bestAction = random.choice(actions)
      bestScore = sum(actionScores)/float(len(actionScores))
    else:
      bestActionIndex = actionScores.index(max(actionScores))
      bestAction = actions[bestActionIndex]
      bestScore = actionScores[bestActionIndex]
    return bestScore, bestAction

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    In this method, we implemented a evaluation function that employs a BFS from pacman's new position
    to find current positions of all the ghosts and food. Then the function evaluates 
    pacman's current position with the following rule:
    1. Take the distance from the closest food +1 and divide it by 30 to get food score.
       This makes sure the closer pacman is to a food the higher the score will be. +1 prevents zero division.
    2. Find the closest ghost and check the scared times. Divide 300 by its distance from the pacman +1.
       If this ghost is scared, this ghost score is positive. If the ghost is not scared. score is negative.
       If the closest ghost is more than 3 units away from the pacman, ghost score is 0.
       This makes sure that the pacman avoids the ghost at a closer distance by making the weight large (300/dist) 
       and also makes sure pacman can be fearless when the ghost is scared.
    3. Add the scared time up as the scared time. This takes the remaining scared time of the ghosts into consideration
       when evaluating pacman's position.

    4. Add the previous 3 scores and the next position's getScore() together and return the value as pacman's current evaluation. 
    
  """
  "*** YOUR CODE HERE ***"
  successorGameState = currentGameState
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

  food_score = 30/((min(food_dists) if len(food_dists) > 0 else 0)+1)
  newScore = 0 +food_score - ghost_score +sum(newScaredTimes) + successorGameState.getScore()
  return newScore

# Abbreviation
better = betterEvaluationFunction

def Debug(category, messages):
  #print "Debug: " + str(category) + ": " + " ".join([str(m) for m in messages]) #TODO remove this before grading
  pass

def Warning(category, messages):
  print "Warning: " + str(category) + ": " + " ".join([str(m) for m in messages])

# GameLevel
# Stores static information of the level, including walls, capsules, food, pacman start position and ghost respawn points
class GameLevel:
  WALL = 0
  EMPTY = 1
  FOOD = 2
  CAPSULE = 3
  def __init__(self, gameState):
    foodMap = gameState.getFood()
    wallMap = gameState.getWalls()
    capsules = gameState.getCapsules()
    self.width_ = foodMap.width
    self.height_ = foodMap.height
    self.data_ = {}
    for x in range(self.width_):
      for y in range(self.height_):
        if wallMap[x][y]:
          self.data_[(x, y)] = GameLevel.WALL
        elif foodMap[x][y]:
          self.data_[(x, y)] = GameLevel.FOOD
        elif (x, y) in capsules:
          self.data_[(x, y)] = GameLevel.CAPSULE
        else:
          self.data_[(x, y)] = GameLevel.EMPTY
    self.pacmanPosition_ = gameState.getPacmanPosition()
    self.ghostPositions_ = gameState.getGhostPositions()
    Debug("GameLevel-init", ("size of the maze is", self.width(), self.height()))
    Debug("GameLevel-init", ("number of food is", len(self.findAll(GameLevel.FOOD))))
    Debug("GameLevel-init", ("number of capsules is", len(self.findAll(GameLevel.CAPSULE))))
    Debug("GameLevel-init", ("pacman position is", self.pacmanPosition()))
    Debug("GameLevel-init", ("ghost positions are", self.ghostPositions()))
  def findAll(self, type):
    return [k for k, v in self.data_.items() if v == type]
  def width(self):
    return self.width_
  def height(self):
    return self.height_
  def data(self):
    return self.data_
  def pacmanPosition(self):
    return self.pacmanPosition_
  def ghostPositions(self):
    return self.ghostPositions_

# Graph
# Provides the data structure to store and manipulate graphs with vertices and directional edges
class Graph:
  def __init__(self):
    self.vertices_ = []
    self.edges_ = []
    self.handleLists_ = []
  def copy(self):
    g = Graph()
    g.vertices_ = self.vertices_[:]
    g.edges_ = self.edges_[:]
    g.handleLists_ = [hl[:] for hl in self.handleLists_]
    return g
  def addVertex(self): # returns vertex number
    vNum = len(self.vertices_)
    self.vertices_.append(())
    self.handleLists_.append([])
    return vNum
  def addEdge(self, vNumBegin, vNumEnd): # returns edge number
    eNum = len(self.edges_)
    hNumBegin = len(self.handleLists_[vNumBegin])
    hNumEnd = len(self.handleLists_[vNumEnd])
    self.edges_.append((vNumBegin, vNumEnd, hNumBegin, hNumEnd))
    self.handleLists_[vNumBegin].append((eNum, 0))
    self.handleLists_[vNumEnd].append((eNum, 1))
    return eNum
  def removeEdge(self, eNum):
    vNumBegin, vNumEnd, hNumBegin, hNumEnd = self.edges_[eNum]
    self.edges_[eNum] = None
    self.handleLists_[vNumBegin][hNumBegin] = None
    self.handleLists_[vNumEnd][hNumEnd] = None
  def removeVertex(self, vNum):
    for i in range(len(self.handleLists_[vNum])):
      if self.handleLists_[vNum][i] is not None:
        eNum, eSide = self.handleLists_[vNum][i]
        self.removeEdge(eNum)
    self.vertices_[vNum] = None
  def shrink(self): # returns oldvnum2newvnum, oldenum2newenum, newvnum2oldvnum, newenum2oldenum
    forwardV = {}
    forwardE = {}
    backwardV = {}
    backwardE = {}
    g = Graph()
    for vNum in range(len(self.vertices_)):
      if self.vertices_[vNum] is not None:
        vNumNew = g.addVertex()
        forwardV[vNum] = vNumNew
        backwardV[vNumNew] = vNum
    for eNum in range(len(self.edges_)):
      if self.edges_[eNum] is not None:
        vNumBegin, vNumEnd, hNumBegin, hNumEnd = self.edges_[eNum]
        eNumNew = g.addEdge(forwardV[vNumBegin], forwardV[vNumEnd])
        forwardE[eNum] = eNumNew
        backwardE[eNumNew] = eNum
    self.vertices_ = g.vertices_
    self.edges_ = g.edges_
    self.handleLists_ = g.handleLists_
    return forwardV, forwardE, backwardV, backwardE
  def validVertices(self):
    return [i for i in range(len(self.vertices_)) if self.vertices_[i] is not None]
  def validEdges(self):
    return [i for i in range(len(self.edges_)) if self.edges_[i] is not None]
  def verticesForEdge(self, eNum):
    vNumBegin, vNumEnd, hNumBegin, hNumEnd = self.edges_[eNum]
    return vNumBegin, vNumEnd
  def handleListForVertex(self, vNum):
    return [h for h in self.handleLists_[vNum] if h is not None]
  def edgeAndSideForHandle(self, handle):
    return handle
  def nextVertexForHandle(self, handle):
    eNum, eSide = handle
    return self.edges_[eNum][1 - eSide]

# TopologicalMap
# Defines a compact representation of the game state that encodes the deterministic part of the game without the ghosts
class TopologicalMap:
  def __init__(self, gameLevel):
    width = gameLevel.width()
    height = gameLevel.height()
    data = gameLevel.data()
    initialPosition = gameLevel.pacmanPosition()
    # graph
    graph = Graph()
    positionForVertex = {}
    vertexForPosition = {}
    moveForEdge = {}
    for x in range(width):
      for y in range(height):
        if data[(x, y)] != GameLevel.WALL:
          vertex = graph.addVertex()
          positionForVertex[vertex] = (x, y)
          vertexForPosition[(x, y)] = vertex
    for x in range(width - 1):
      for y in range(height):
        if (x, y) in vertexForPosition and (x + 1, y) in vertexForPosition:
          edge = graph.addEdge(vertexForPosition[(x, y)], vertexForPosition[(x + 1, y)])
          moveForEdge[edge] = ((1, 0), (-1, 0))
    for x in range(width):
      for y in range(height - 1):
        if (x, y) in vertexForPosition and (x, y + 1) in vertexForPosition:
          edge = graph.addEdge(vertexForPosition[(x, y)], vertexForPosition[(x, y + 1)])
          moveForEdge[edge] = ((0, 1), (0, -1))
    # food
    edgeContainsFood = {e: False for e in graph.validEdges()}
    tmpFoodFlags = {v: data[positionForVertex[v]] == GameLevel.FOOD for v in graph.validVertices()}
    for edge in graph.validEdges():
      v1, v2 = graph.verticesForEdge(edge)
      if tmpFoodFlags[v1] and tmpFoodFlags[v2]:
        edgeContainsFood[edge] = True
        tmpFoodFlags[v1] = False
        tmpFoodFlags[v2] = False
    for edge in graph.validEdges():
      v1, v2 = graph.verticesForEdge(edge)
      if tmpFoodFlags[v1] or tmpFoodFlags[v2]:
        edgeContainsFood[edge] = True
        tmpFoodFlags[v1] = False
        tmpFoodFlags[v2] = False
    Debug("TopologicalMap-init", ("original graph has", len(graph.validVertices()), "vertices", len(graph.validEdges()), "edges"))
    # capsules
    vertexIsCapsule = {v: data[positionForVertex[v]] == GameLevel.CAPSULE for v in graph.validVertices()}
    # hardcoded break points
    vertexIsBreakPoint = {v: positionForVertex[v] in [(None, 15, 2), (18, 6), (18, 4), (18, 3), (15, 6), (15, 7), (16, 4), (16, 3), (9, 1), (None, 13, 6), (None, 13, 2), (None, 1, 3), (4, 7)] for v in graph.validVertices()}
    # path graph
    pathSequences = {e: [(e, False)] for e in graph.validEdges()}
    pathGraph = graph.copy()
    for vertex in pathGraph.validVertices():
      handleList = pathGraph.handleListForVertex(vertex)
      if len(handleList) != 2 or vertexIsCapsule[vertex] or vertexIsBreakPoint[vertex]:
        continue
      edge1, side1 = pathGraph.edgeAndSideForHandle(handleList[0])
      edge2, side2 = pathGraph.edgeAndSideForHandle(handleList[1])
      if edge1 == edge2:
        continue
      vertex1 = pathGraph.nextVertexForHandle(handleList[0])
      vertex2 = pathGraph.nextVertexForHandle(handleList[1])
      edge = pathGraph.addEdge(vertex1, vertex2)
      pathGraph.removeEdge(edge1)
      pathGraph.removeEdge(edge2)
      pathGraph.removeVertex(vertex)
      sequence1 = pathSequences[edge1][:]
      if not side1:
        sequence1.reverse()
        sequence1 = [(e, not s) for e, s in sequence1]
      sequence2 = pathSequences[edge2][:]
      if side2:
        sequence2.reverse()
        sequence2 = [(e, not s) for e, s in sequence2]
      pathSequences[edge] = sequence1 + sequence2
    Debug("TopologicalMap-init", ("path graph has", len(pathGraph.validVertices()), "vertices", len(pathGraph.validEdges()), "edges"))
    forwardV, forwardE, backwardV, backwardE = pathGraph.shrink()
    pathSequences = {pe: pathSequences[peOld] for pe, peOld in backwardE.items()}
    originalVertices = {pv: v for pv, v in backwardV.items()}
    pathVertexForVertex = {v: forwardV[v] if v in forwardV else None for v in graph.validVertices()}
    pathEdgeContainsFood = {pe: False for pe in pathGraph.validEdges()}
    edgeLocations = {}
    movesForPathEdge = {}
    for pathEdge in pathGraph.validEdges():
      moves = ([], [])
      for index, (edge, side) in enumerate(pathSequences[pathEdge]):
        moves[0].append(moveForEdge[edge][1 if side else 0])
        moves[1].append(moveForEdge[edge][0 if side else 1])
        if edgeContainsFood[edge]:
          pathEdgeContainsFood[pathEdge] = True
        edgeLocations[edge] = (pathEdge, index)
      moves[1].reverse()
      movesForPathEdge[pathEdge] = (tuple(moves[0]), tuple(moves[1]))
    # food distances and capsule distances
    foodLocations = [pe for pe in pathGraph.validEdges() if pathEdgeContainsFood[pe]]
    capsuleLocations = [pv for pv in pathGraph.validVertices() if vertexIsCapsule[originalVertices[pv]]]
    pathVertexDistanceMatrix = {}
    for basePathVertex in pathGraph.validVertices():
      distanceMap = {}
      ucsQueue = util.PriorityQueue()
      ucsQueue.push((basePathVertex, 0), 0)
      while not ucsQueue.isEmpty():
        pathVertex, distance = ucsQueue.pop()
        if pathVertex in distanceMap:
          continue
        distanceMap[pathVertex] = distance
        for handle in pathGraph.handleListForVertex(pathVertex):
          nextPathVertex = pathGraph.nextVertexForHandle(handle)
          pathEdge, pathSide = pathGraph.edgeAndSideForHandle(handle)
          pathLength = len(pathSequences[pathEdge])
          ucsQueue.push((nextPathVertex, distance + pathLength), distance + pathLength)
        for pv in pathGraph.validVertices():
          if pv not in distanceMap:
            distanceMap[pv] = 1000 #TODO: remove this
      pathVertexDistanceMatrix[basePathVertex] = distanceMap
    pathVertexFoodDistances = {}
    pathVertexCapsuleDistances = {}
    for pathVertex in pathGraph.validVertices():
      pathVertexFoodDistances[pathVertex] = [min(pathVertexDistanceMatrix[pathVertex][pv] for pv in pathGraph.verticesForEdge(pe)) for pe in foodLocations]
      pathVertexCapsuleDistances[pathVertex] = [pathVertexDistanceMatrix[pathVertex][pv] for pv in capsuleLocations]
    # instance variables
    self.initialPosition_ = initialPosition
    self.capsuleLocations_ = capsuleLocations
    self.capsuleForPathVertex_ = [self.capsuleLocations_.index(pv) if pv in self.capsuleLocations_ else None for pv in pathGraph.validVertices()]
    self.foodLocations_ = foodLocations
    self.foodForPathEdge_ = [self.foodLocations_.index(pe) if pe in foodLocations else None for pe in pathGraph.validEdges()]
    Debug("TopologicalMap-init", (len(self.capsuleLocations_), "capsule locations"))
    Debug("TopologicalMap-init", (len(self.foodLocations_), "food locations"))
    self.movesForPathEdge_ = movesForPathEdge
    self.vertexForPosition_ = vertexForPosition
    self.pathVertexForVertex_ = pathVertexForVertex
    self.pathGraph_ = pathGraph
    self.positionForPathVertex_ = {pv: positionForVertex[backwardV[pv]] for pv in pathGraph.validVertices()}
    self.moveForEdge_ = moveForEdge
    self.graph_ = graph
    self.pathVertexFoodDistances_ = pathVertexFoodDistances
    self.pathVertexCapsuleDistances_ = pathVertexCapsuleDistances
  def distanceToAnyFood(self, state):
    position, pathVertex, capsules, food = state
    if pathVertex is None:
      for nextState, movesId, movesStart, moves, hasCapsule, hasFood in self.successors(state):
        pathVertex = nextState[1]
        break
    distances = self.pathVertexFoodDistances_[pathVertex]
    if sum(food) == 0:
      return 0
    return min(d for fl, d in zip(food, distances) if fl)
  def distanceToAnyCapsules(self, state):
    position, pathVertex, capsules, food = state
    if pathVertex is None:
      for nextState, movesId, movesStart, moves, hasCapsule, hasFood in self.successors(state):
        pathVertex = nextState[1]
        break
    if sum(capsules) == 0:
      return 0
    distances = self.pathVertexCapsuleDistances_[pathVertex]
    return min(d for fl, d in zip(capsules, distances) if fl)
  def foodCountForState(self, state):
    position, pathVertex, capsules, food = state
    return sum(food)
  def capsuleCountForState(self, state):
    position, pathVertex, capsules, food = state
    return sum(capsules)
  def initialState(self): # returns state=(position=(x, y), path vertex or none, capsule flags, food flags)
    position = self.initialPosition_
    vertex = self.vertexForPosition_[position]
    pathVertex = self.pathVertexForVertex_[vertex]
    capsules = [True for pv in self.capsuleLocations_]
    if pathVertex is not None and self.capsuleForPathVertex_[pathVertex] is not None:
      capsules[self.capsuleForPathVertex_[pathVertex]] = False
    food = [True for pe in self.foodLocations_]
    return position, pathVertex, capsules, food
  def successors(self, state): # returns [(state, movesId, movesStart, moves=((dx, dy), ...), has capsule, has food), ...]
    position, pathVertex, capsules, food = state
    if pathVertex is not None:
      successors = []
      for handle in self.pathGraph_.handleListForVertex(pathVertex):
        nextPathVertex = self.pathGraph_.nextVertexForHandle(handle)
        pathEdge, pathSide = self.pathGraph_.edgeAndSideForHandle(handle)
        nextPosition = self.positionForPathVertex_[nextPathVertex]
        nextCapsules = capsules[:]
        capsuleIndex = self.capsuleForPathVertex_[nextPathVertex]
        if capsuleIndex is not None and nextCapsules[capsuleIndex]:
          nextCapsules[capsuleIndex] = False
          hasCapsule = True
        else:
          hasCapsule = False
        nextFood = food[:]
        foodIndex = self.foodForPathEdge_[pathEdge]
        if foodIndex is not None and nextFood[foodIndex]:
          nextFood[foodIndex] = False
          hasFood = True
        else:
          hasFood = False
        movesId = (pathEdge, pathSide)
        moves = self.movesForPathEdge_[pathEdge][1 if pathSide else 0]
        successors.append(((nextPosition, nextPathVertex, nextCapsules, nextFood), movesId, position, moves, hasCapsule, hasFood))
      return successors
    Debug("TopologicalMap-successors", ("vertex not on the path graph",))
    successors = []
    vertex = self.vertexForPosition_[position]
    for handle in self.graph_.handleListForVertex(vertex):
      movesId = None
      moves = []
      e, s = self.graph_.edgeAndSideForHandle(handle)
      moves.append(self.moveForEdge_[e][s])
      prev = vertex
      v = self.graph_.nextVertexForHandle(handle)
      while self.pathVertexForVertex_[v] is None:
        for h in self.graph_.handleListForVertex(v):
          if self.graph_.nextVertexForHandle(h) != prev:
            prev = v
            v = self.graph_.nextVertexForHandle(h)
            e, s = self.graph_.edgeAndSideForHandle(h)
            moves.append(self.moveForEdge_[e][s])
            break
      nextPathVertex = self.pathVertexForVertex_[v]
      nextPosition = self.positionForPathVertex_[nextPathVertex]
      nextCapsules = capsules[:]
      capsuleIndex = self.capsuleForPathVertex_[nextPathVertex]
      if capsuleIndex is not None and nextCapsules[capsuleIndex]:
        nextCapsules[capsuleIndex] = False
        hasCapsule = True
      else:
        hasCapsule = False
      nextFood = food[:]
      hasFood = False
      successors.append(((nextPosition, nextPathVertex, nextCapsules, nextFood), movesId, position, moves, hasCapsule, hasFood))
    return successors
  def findAllCanonicalActions(): # returns [(movesId, movesStart, moves), ...]
    result = []
    for pathVertex in self.pathGraph_.validVertices():
      startPosition = self.positionForPathVertex_(pathVertex)
      for handle in self.pathGraph_.handleListForVertex(pathVertex):
        pathEdge, pathSide = self.pathGraph_.edgeAndSideForHandle(handle)
        movesId = (pathEdge, pathSide)
        moves = self.movesForPathEdge_[pathEdge][1 if pathSide else 0]
        result.append((movesId, startPosition, moves))
    return result
  def validateStateAgainstGameState(self, state, gameState):
    position, pathVertex, capsules, food = state
    if gameState.getPacmanPosition() != position:
      return False
    if pathVertex != self.pathVertexForVertex_[self.vertexForPosition_[position]]:
      return False
    myCapsules = [self.positionForPathVertex_[self.capsuleLocations_[i]] for i in range(len(capsules)) if capsules[i]]
    if len(myCapsules) != sum(capsules) or len(myCapsules) != len(gameState.getCapsules()):
      return False
    if any(c not in myCapsules for c in gameState.getCapsules()):
      return False
    foodMap = gameState.getFood().copy()
    width = foodMap.width
    height = foodMap.height
    for i in range(len(food)):
      if food[i]:
        pathEdge = self.foodLocations_[i]
        pathVertex = self.pathGraph_.verticesForEdge(pathEdge)[0]
        moves = self.movesForPathEdge_[pathEdge][0]
        x, y = self.positionForPathVertex_[pathVertex]
        foodMap[x][y] = False
        for dx, dy in moves:
          x += dx
          y += dy
          foodMap[x][y] = False
    if foodMap.count() != 0:
      return False
    return True

class GhostBehaviours:
  def __init__(self, gameLevel):
    self.width_ = gameLevel.width()
    self.height_ = gameLevel.height()
    self.levelData_ = gameLevel.data()
    self.initialGhostPositions_ = gameLevel.ghostPositions()
    self.cache_ = {}
    self.isInteger_ = {(x, y): True for x in range(self.width_) for y in range(self.height_)}
    self.neighborOffsets_ = ((-1, 0), (0, -1), (0, 1), (1, 0))
  def ghostMove(self, position, ghostPosition, ghostFacing, isScared, allowRandom):
    # returns [(probability, newPosition, newFacing)]
    if ghostPosition not in self.isInteger_:
      newPosition = (ghostPosition[0] + 0.5 * ghostFacing[0], ghostPosition[1] + 0.5 * ghostFacing[1])
      if newPosition not in self.isInteger_ or not isScared:
        Warning("GhostBehaviours-ghostMove", ("invalid coordinates", ghostPosition, newPosition, "scared", isScared))
      return [(1.0, newPosition, ghostFacing)]
    else:
      if isScared:
        idealDx = cmp(ghostPosition[0], position[0])
        idealDy = cmp(ghostPosition[1], position[1])
        speed = 0.5
      else:
        idealDx = cmp(position[0], ghostPosition[0])
        idealDy = cmp(position[1], ghostPosition[1])
        speed = 1.0
      newFacings = []
      newPositions = []
      costs = []
      for dx, dy in self.neighborOffsets_:
        if ghostFacing == (-dx, -dy):
          continue
        if self.levelData_[(ghostPosition[0] + dx, ghostPosition[1] + dy)] != GameLevel.WALL:
          newFacings.append((dx, dy))
          newPositions.append((ghostPosition[0] + speed * dx, ghostPosition[1] + speed * dy))
          costs.append(abs(dx - idealDx) + abs(dy - idealDy))
      if len(newFacings) > 0:
        minCost = min(costs)
        bestCount = costs.count(minCost)
        feasibleCount = len(newFacings)
        if allowRandom:
          probabilities = [0.2 / feasibleCount + (0.8 / bestCount if c == minCost else 0.0) for c in costs]
        else:
          indices = [i for i in range(feasibleCount) if costs[i] == minCost]
          newFacings = [newFacings[i] for i in indices]
          newPositions = [newPositions[i] for i in indices]
          costs = [costs[i] for i in indices]
          probabilities = [1.0 / bestCount] * bestCount
      else:
        newFacings.append((-ghostFacing[0], -ghostFacing[1]))
        newPositions.append((ghostPosition[0] - speed * ghostFacing[0], ghostPosition[1] - speed * ghostFacing[1]))
        probabilities = [1.0]
      if abs(sum(probabilities) - 1.0) > 0.001:
        Warning("GhostBehaviours-ghostMove", ("sum of probabilities is", sum(probabilities)));
      return [(probabilities[i], newPositions[i], newFacings[i]) for i in range(len(newFacings)) if probabilities[i] > 0]
  def clearCacheStatistics(self):
    self.statTotal_ = 0
    self.statCacheable_ = 0
    self.statCacheHit_ = 0
  def reportCacheStatistics(self, cacheable, cacheHit):
    self.statTotal_ += 1
    if cacheable:
      self.statCacheable_ += 1
    if cacheHit:
      self.statCacheHit_ += 1
    if self.statTotal_ == 10000:
      Debug("GhostBehaviours-reportCacheStatistics", ("hit", self.statCacheHit_ / 10000.0, "cachable", self.statCacheable_ / 10000.0, "size", len(self.cache_)));
      self.clearCacheStatistics()
  def computeGhostBehaviour(self, movesId, movesStart, moves, hasCapsuleInEnd, ghostIndex, ghostState, branchingThreshold):
    # ghostState=(ghostPosition, ghostFacing, scaredTime)
    # returns [(probability, will score, will lose, newGhostState=(newGhostPosition, newGhostFacing, newScaredTime)), ...]
    cacheKey = None
    if movesId is not None:
      if ghostState[2] >= len(moves) + 1:
        originalScaredTime = ghostState[2] # so that we can compute the relative scared time; this is ugly
        cacheKey = (movesId, ghostState[0], ghostState[1], ghostIndex, hasCapsuleInEnd, -1, branchingThreshold)
      else:
        originalScaredTime = None # don't use the relative scared time
        cacheKey = (movesId, ghostState[0], ghostState[1], ghostIndex, hasCapsuleInEnd, ghostState[2], branchingThreshold)
    if cacheKey is not None and cacheKey in self.cache_:
      self.reportCacheStatistics(True, True)
      if originalScaredTime is None:
        return self.cache_[cacheKey]
      else:
        distribution = []
        for probability, willScore, willLose, (newGhostPosition, newGhostFacing, newScaredTime) in self.cache_[cacheKey]:
          if newScaredTime < 0: # the store scared time is a relative scared time
            newScaredTime += originalScaredTime
            if newScaredTime == 0: # this is ugly
              newGhostPosition = util.nearestPoint(newGhostPosition)
          distribution.append((probability, willScore, willLose, (newGhostPosition, newGhostFacing, newScaredTime)))
        return distribution
    pacmanPosition = movesStart
    distribution = [(1.0, False, False, ghostState)]
    for moveIndex in range(len(moves)):
      isEatingCapsule = (hasCapsuleInEnd and moveIndex == len(moves) - 1)
      pacmanPosition = (pacmanPosition[0] + moves[moveIndex][0], pacmanPosition[1] + moves[moveIndex][1])
      # the pacman has moved, compute the consequences
      for i in range(len(distribution)):
        probability, willScore, willLose, ghostState = distribution[i]
        if isEatingCapsule:
          ghostState = (ghostState[0], ghostState[1], 40) # the duration of the capsule is 40
        isScared = (ghostState[2] > 0)
        if abs(pacmanPosition[0] - ghostState[0][0]) < 1 and abs(pacmanPosition[1] - ghostState[0][1]) < 1:
          if isScared:
            willScore = True
            ghostState = (self.initialGhostPositions_[ghostIndex], (0, 0), 0) # revive the ghost
          else:
            willLose = True
        distribution[i] = (probability, willScore, willLose, ghostState)
      # now move the ghost
      nextDistribution = []
      for probability, willScore, willLose, ghostState in distribution:
        isScared = ghostState[2] > 0 # use the old scared time to decide the move
        for subProbability, newPosition, newFacing in self.ghostMove(pacmanPosition, ghostState[0], ghostState[1], isScared, probability > branchingThreshold):
          subWillScore = willScore
          subWillLose = willLose
          subGhostState = (newPosition, newFacing, max(0, ghostState[2] - 1))
          if ghostState[2] == 1:
            subGhostState = (util.nearestPoint(newPosition), newFacing, 0)
          isScared = subGhostState[2] > 0 # use the new scared time to decide the result of collision result
          if abs(pacmanPosition[0] - newPosition[0]) < 1 and abs(pacmanPosition[1] - newPosition[1]) < 1:
            if isScared:
              subWillScore = True
              subGhostState = (self.initialGhostPositions_[ghostIndex], (0, 0), 0) # revive the ghost
            else:
              subWillLose = True
          nextDistribution.append((probability * subProbability, subWillScore, subWillLose, subGhostState))
      distribution = nextDistribution
    if cacheKey is not None:
      self.reportCacheStatistics(True, False)
      if originalScaredTime is None:
        self.cache_[cacheKey] = distribution
      else:
        relativeDistribution = []
        for probability, willScore, willLose, (newGhostPosition, newGhostFacing, newScaredTime) in distribution:
          if not hasCapsuleInEnd and not willScore:
            newScaredTime -= originalScaredTime
            if newScaredTime >= 0:
              Warning("GhostBehaviours-computeGhostBehaviour", ("relative scared time greater than zero ", newScaredTime, originalScaredTime))
          relativeDistribution.append((probability, willScore, willLose, (newGhostPosition, newGhostFacing, newScaredTime)))
        self.cache_[cacheKey] = relativeDistribution
    else:
      self.reportCacheStatistics(False, False)
    return distribution
  def ghostStatesForCurrentGameState(self, gameState):
    ghostStates = []
    for agentState in gameState.getGhostStates():
      ghostPosition = agentState.getPosition()
      ghostFacing = Actions.directionToVector(agentState.getDirection())
      scaredTime = agentState.scaredTimer
      ghostStates.append((ghostPosition, ghostFacing, scaredTime))
    return ghostStates

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """
  def __init__(self):
    self.initialized = False
  def findRandomSuccessor(self, gameState):
    currentState = self.state
    topologicalMap = self.topologicalMap
    ghostBehaviours = self.ghostBehaviours
    currentPosition, currentPathVertex, currentCapsules, currentFood = currentState
    
    if not topologicalMap.validateStateAgainstGameState(currentState, gameState):
      Warning("ContestAgent-findBestSuccessor", ("state validation failed, state is", currentState))
    
    currentGhostStates = ghostBehaviours.ghostStatesForCurrentGameState(gameState)
    print "---------------- current ghost0 state ------------------"
    print currentGhostStates[0]
    print "----------------------------------"
    
    successors = topologicalMap.successors(currentState)
    successor = successors[random.randrange(len(successors))]
    
    nextState, movesId, movesStart, moves, hasCapsule, hasFood = successor
    nextPosition, nextPathVertex, nextCapsules, nextFood = nextState
    
  #def computeGhostBehaviour(self, movesId, movesStart, moves, hasCapsuleInEnd, ghostIndex, ghostState):
    # ghostState=(ghostPosition, ghostFacing, scaredTime)
    # returns [(probability, will score, will lose, newGhostState=(newGhostPosition, newGhostFacing, newScaredTime)), ...]
    print "---------------- predicted next ghost0 states ------------------"
    for probability, willScore, willLose, nextGhostState in (
        ghostBehaviours.computeGhostBehaviour(movesId, movesStart, moves, hasCapsule, 0, currentGhostStates[0], 0)):
      print probability, willScore, willLose
      print nextGhostState
    print "----------------------------------"
    
    return successor
  def combineGhostDistributions(self, distributions):
    # TODO separate each ghosts to reduce exponential explosion of tree size
    #returns distribution=[(probability, score, willLose, newGhostStates), ...], death probability
     # the death probability is more accurate than the distribution
    lengths = [len(d) for d in distributions]
    combinations = [()]
    for i in range(len(lengths)):
      combinations = [c + (j,) for c in combinations for j in range(lengths[i])]
    distribution = []
    deathProbability = 0.0
    for indices in combinations:
      probability = 1.0
      score = 0.0
      willLose = False
      newGhostStates = []
      for i, j in enumerate(indices):
        case = distributions[i][j]
        probability *= case[0]
        score += case[1]
        willLose = willLose or case[2]
        newGhostStates.append(case[3])
      if willLose:
        deathProbability += probability
      if probability > 0.05:
        distribution.append((probability, score, willLose, newGhostStates))
    distribution.sort(reverse=True)
    if len(distribution) > 8: # should be no less than 8, so that the pacman won't ignore the risks
      threshold = distribution[7][0]
      distribution = [d for d in distribution if d[0] >= threshold - 0.001]
      scale = 1.0 / sum(d[0] for d in distribution)
      distribution = [(d[0] * scale,) + d[1:] for d in distribution]
    return distribution, deathProbability
  def evaluateStateRecursive(self, topologicalMap, ghostBehaviours, state, ghostStates, currentDepth, currentProbability):
    # returns best value, best successor
    ghostCount = len(ghostStates)
    successors = topologicalMap.successors(state)
    successorValues = []
    capsuleCount = sum(state[2])
    foodCount = sum(state[3])
    totalScaredTime = sum(s[2] for s in ghostStates)
    if topologicalMap.foodCountForState(state) == 0:
      if capsuleCount > 1 or capsuleCount == 1 and totalScaredTime > 20:
        return -500, None
      elif capsuleCount == 1:
        return 100, None
      elif totalScaredTime > 20:
        return 400, None
      else:
        return 700, None
    if (0.3 ** currentDepth) * currentProbability < 0.00003:
      #TODO: no death in small steps != no death/better value
      value = 0
      value += 1.0 * totalScaredTime
      value += 40.0 * (0.5 ** (topologicalMap.distanceToAnyCapsules(state) / 10.0))
      value += 5.0 * (0.5 ** (topologicalMap.distanceToAnyFood(state) / 5.0))
      return value, None
    for nextState, movesId, movesStart, moves, hasCapsule, hasFood in successors:
      successorValue = 0
      branchingThreshold = 0.3 if currentProbability > 0.05 else 1.0
      distributions = [ghostBehaviours.computeGhostBehaviour(movesId, movesStart, moves, hasCapsule, i, ghostStates[i], branchingThreshold) for i in range(ghostCount)]
      distribution, deathProbability = self.combineGhostDistributions(distributions)
      if hasFood:
        successorValue += 30
      else:
        successorValue -= len(moves)
      if hasCapsule:
        if totalScaredTime > 20:
          successorValue -= 300
        else:
          successorValue -= 100
      successorValue -= deathProbability * 8000
      if self.trackingId == 4999 and currentDepth == 0:
        print distributions
        print distribution, deathProbability
      for probability, score, willLose, nextGhostStates in distribution:
        caseValue = 0
        caseValue += score * 300
        if not willLose and deathProbability < 0.3: # if this route is dangerous enough, stop considering anything further
          #TODO use better pruning logic
          caseValue += (0.98 ** len(moves)) * self.evaluateStateRecursive(topologicalMap, ghostBehaviours, nextState, nextGhostStates, currentDepth + 1, currentProbability * probability)[0]
        successorValue += probability * caseValue
      successorValues.append(successorValue)
    if self.trackingId == 4999 and currentDepth == 0:
      print successors
      print successorValues
    value = max(successorValues)
    candidates = [successors[i] for i in range(len(successors)) if successorValues[i] >= value - 0.001]
    successor = candidates[random.randrange(len(candidates))]
    return value, successor
  def findBestSuccessor(self, gameState):
    topologicalMap = self.topologicalMap
    ghostBehaviours = self.ghostBehaviours
    state = self.state
    if not topologicalMap.validateStateAgainstGameState(state, gameState):
      Warning("ContestAgent-findBestSuccessor", ("state validation failed, state is", state))
    ghostStates = ghostBehaviours.ghostStatesForCurrentGameState(gameState)
    currentGhostStates = ghostBehaviours.ghostStatesForCurrentGameState(gameState)
    value, successor = self.evaluateStateRecursive(topologicalMap, ghostBehaviours, state, ghostStates, 0, 1.0)
    if False:
      print "----------------- current states -----------------"
      print state
      print currentGhostStates
      print "----------------- successor chosen -----------------"
      print successor
      print "---------------- predicted states ------------------"
      nextState, movesId, movesStart, moves, hasCapsule, hasFood = successor
      for i in range(len(currentGhostStates)):
        print "ghost" + str(i)
        for probability, willScore, willLose, nextGhostState in (
            ghostBehaviours.computeGhostBehaviour(movesId, movesStart, moves, hasCapsule, i, currentGhostStates[i], 3)):
          print probability, willScore, willLose
          print nextGhostState
      print "----------------------------------"
    Debug("ContestAgent-findBestSuccessor", ("value =", value, "tracking id", self.trackingId))
    self.trackingId += 1
    return successor
  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    
    """
    Description:
    In the mini-contest problem, we implemented a context agent that utilizes expectimax algorithm with depth of 
    variable depth (lower depth toward low possibility event) by buildiing a graph and a cache for ghost's
    actions.
    1. The graph makes the assumption that pacman will not go backward once it is within a pipe (no exit between the two end points). 
       This efficiently decreases the searching scope and minizes pacman's decision making process. We also discard those event that have
       very low chance of happening to trim nodes in the searching process to make it faster.
    2. For each possible successsor state, we evaluate it based on several different factors: food, ghost's scare time, pellet position,
       and the winning/loss chances. A score will be calculated for the current state and we use a decay factor to prevent replanning as well
       as truthfully evaluting the reward of future actions. We only evaluate new actions when pacman reachs a split point, so that he will
       move swiftly in a pipe.
    3. To maximize our score in this game, we applied the following tricks:
        a) Pacman will not eat the last food if there is still pellet remaining. He will go for the pellet and eat as many ghosts as possible 
        b) We give pellet a bigger score when there is ghost chasing the pacman.
        c) We set the score of eating a pellet to be negative -100 and -300 to make sure that it is only used when eating ghosts are possible,
           so that the pellets will not be wasted.
        
    """
    
    "*** YOUR CODE HERE ***"
    if not self.initialized:
      self.initialized = True
      self.gameLevel = GameLevel(gameState)
      self.topologicalMap = TopologicalMap(self.gameLevel)
      self.ghostBehaviours = GhostBehaviours(self.gameLevel)
      self.state = self.topologicalMap.initialState()
      self.pendingMoves = []
      self.trackingId = 0
      self.time = 0
    if gameState.getPacmanState().getDirection() == Directions.STOP:
      Debug("ContestAgent-getAction", ("resetting...",))
      self.ghostBehaviours.clearCacheStatistics()
      self.state = self.topologicalMap.initialState()
      self.pendingMoves = []
    if len(self.pendingMoves) == 0:
      state, movesId, moveStart, moves, hasCapsule, hasFood = self.findBestSuccessor(gameState)
      self.state = state
      self.pendingMoves = moves
    move = self.pendingMoves[0]
    self.pendingMoves = self.pendingMoves[1:]
    action = self.actionForMove(move)
    return action
  def actionForMove(self, move):
    return Actions.vectorToDirection(move)
