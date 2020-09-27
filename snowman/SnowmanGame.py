from Game import Game
import numpy as np

SNOW_PLANE = -1
LAYER_COUNT = 3
PLANE_COUNT = 2 * LAYER_COUNT + 1
MOVEMENTS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
MOVEMENT_COUNT = len(MOVEMENTS)
PLAYER_LAYER_SYMBOLS = ['o', 'C', 'O']
OPPONENT_LAYER_SYMBOLS = ['x', 'Y', 'X']
PLAYER_SNOWMEN_SYMBOLS = ['8', 'S']
OPPONENT_SNOWMEN_SYMBOLS = ['Z', '7']

class SnowmanGame(Game):
  def __init__(self, boardLength):
    self.boardLength = boardLength
    self.boardSize = (PLANE_COUNT, self.boardLength, self.boardLength)
    self.actionSize = boardLength * boardLength * MOVEMENT_COUNT

  def getInitBoard(self):
    board = np.zeros(self.boardSize, dtype=int)
    board[SNOW_PLANE] = 1 # all covered in snow
    return board

  def getBoardSize(self):
    return self.boardSize

  def getActionSize(self):
    return self.actionSize

  def getNextState(self, board, player, action):
    nextBoard = np.copy(board)
    self.executeAction(nextBoard, player, action)
    return (nextBoard, -player)

  def getValidMoves(self, board, player, moveDetection=False):
    valids = None
    if not moveDetection:
      valids = np.zeros(self.actionSize, dtype=int)
    firstLayerPlane = (0 if player == 1 else LAYER_COUNT)
    for y in range(self.boardLength):
      for x in range(self.boardLength):
        for i in range(MOVEMENT_COUNT):
          dx, dy = MOVEMENTS[i]
          targetX = x + dx
          targetY = y + dy
          if (targetX < 0 or targetX >= self.boardLength or
              targetY < 0 or targetY >= self.boardLength):
            continue
          targetWithSnow = (board[SNOW_PLANE][targetY][targetX] != 0)
          if board[SNOW_PLANE][y][x] != 0:
            if not targetWithSnow:
              continue
          else:
            sourceLayerIndex = None
            for j in range(LAYER_COUNT - 1):
              if board[firstLayerPlane + j][y][x] != 0:
                if sourceLayerIndex is not None:
                  sourceLayerIndex = None
                  break
                sourceLayerIndex = j
            if sourceLayerIndex is None:
              continue
            if not targetWithSnow:
              targetWithSnowballs = True
              for j in range(LAYER_COUNT - 1, sourceLayerIndex, -1):
                if board[firstLayerPlane + j][targetY][targetX] == 0:
                  targetWithSnowballs = False
                  break
              if not targetWithSnowballs:
                continue
          if moveDetection:
            return True
          valids[(self.boardLength * y + x) * MOVEMENT_COUNT + i] = 1
    if moveDetection:
      return False
    return valids

  def getGameEnded(self, board, player):
    if self.getValidMoves(board, player, True):
      return 0
    if self.isPlayerWin(board, player):
      return 1
    return -1

  def getCanonicalForm(self, board, player):
    canonicalBoard = np.copy(board)
    if player != 1:
      for i in range(LAYER_COUNT):
        canonicalBoard[i] = board[LAYER_COUNT + i]
        canonicalBoard[LAYER_COUNT + i] = board[i]
    return canonicalBoard

  def getSymmetries(self, board, pi):
    assert len(pi) == self.actionSize
    shapedPi = np.reshape(pi, self.boardSize)
    symmetries = []
    for rotation in range(4):
      for flip in [False, True]:
        if rotation:
          symmetricBoard = np.rot90(board, rotation)
          symmetricPi = np.rot90(shapedPi, rotation)
        else:
          symmetricBoard = board
          symmetricPi = shapedPi
        if flip:
          symmetricBoard = np.fliplr(symmetricBoard)
          symmetricPi = np.fliplr(symmetricPi)
        symmetries += [(symmetricBoard, list(symmetricPi.ravel()))]
    return symmetries

  def stringRepresentation(self, board):
    return board.tostring()

  def executeAction(self, board, player, action):
    assert action >= 0 and action < self.actionSize
    pickAction = int(action / MOVEMENT_COUNT)
    moveDirection = action % MOVEMENT_COUNT
    x = pickAction % self.boardLength
    y = int(pickAction / self.boardLength)
    dx, dy = MOVEMENTS[moveDirection]
    targetX = x + dx
    targetY = y + dy
    assert targetX >= 0 and targetX < self.boardLength
    assert targetY >= 0 and targetY < self.boardLength
    firstLayerPlane = (0 if player == 1 else LAYER_COUNT)
    if board[SNOW_PLANE][y][x] != 0:
      assert board[SNOW_PLANE][targetY][targetX] != 0
      board[SNOW_PLANE][y][x] = 0
      board[SNOW_PLANE][targetY][targetX] = 0
      board[firstLayerPlane][targetY][targetX] = 1
    else:
      sourceLayerIndex = None
      for i in range(LAYER_COUNT - 1):
        if board[firstLayerPlane + i][y][x] != 0:
          assert sourceLayerIndex is None
          sourceLayerIndex = i
      assert sourceLayerIndex is not None
      sourceLayerPlane = firstLayerPlane + sourceLayerIndex
      board[sourceLayerPlane][y][x] = 0
      if board[SNOW_PLANE][targetY][targetX] != 0:
        board[SNOW_PLANE][targetY][targetX] = 0
        assert board[sourceLayerPlane + 1][targetY][targetX] == 0
        board[sourceLayerPlane + 1][targetY][targetX] = 1
      else:
        for i in range(LAYER_COUNT - 1, sourceLayerIndex, -1):
          assert board[firstLayerPlane + i][targetY][targetX] != 0
        board[sourceLayerPlane][targetY][targetX] = 1

  def isPlayerWin(self, board, player):
    firstLayerPlane = (0 if player == 1 else LAYER_COUNT)
    for y in range(self.boardLength):
      for x in range(self.boardLength):
        for i in range(LAYER_COUNT - 1, -1, -1):
          if board[firstLayerPlane + i][y][x] == 0:
            break
          elif i == 0:
            return True
    return False

  # Board example of size 8x8
  # '*' - snow, 'o'/'C'/'O' - snowballs, x'/'Y'/'X' - opponent snowballs
  # 'S'/'8' - snowmens, '7'/'Z' - opponent snowmens
  #     1 2 3 4 5 6 7 8
  #     - - - - - - - -
  # 1 | * * * * * * * * |
  # 2 | * * * *   x * * |
  # 3 |   o * * X Y *   |
  # 4 | *     *     *   |
  # 5 | * C * *       Z |
  # 6 | * * * *   *     |
  # 7 | *       S *     |
  # 8 | * * * *         |
  #     - - - - - - - -
  @staticmethod
  def display(board):
    boardLength = board.shape[-1]
    print("")
    print("  ", end=" ")
    for _ in range(boardLength):
      print("-", end=" ")
    print(" ")
    for y in range(boardLength):
      print(" |",end=" ")
      for x in range(boardLength):
        symbol = None
        if board[SNOW_PLANE][y][x] != 0:
          symbol = "*"
        else:
          for i in range(LAYER_COUNT - 1, -1, -1):
            if board[i][y][x] != 0:
              if symbol is None:
                symbol = PLAYER_LAYER_SYMBOLS[i]
              else:
                symbol = PLAYER_SNOWMEN_SYMBOLS[i]
          for i in range(LAYER_COUNT - 1, -1, -1):
            if board[LAYER_COUNT + i][y][x] != 0:
              if symbol is None:
                symbol = OPPONENT_LAYER_SYMBOLS[i]
              else:
                symbol = OPPONENT_SNOWMEN_SYMBOLS[i]
        if symbol is None:
          symbol = " "
        print(symbol, end=" ")
      print("|")
    print("  ", end=" ")
    for _ in range(boardLength):
      print("-", end=" ")
    print(" ")
    print("")
