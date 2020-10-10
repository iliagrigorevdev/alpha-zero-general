import Arena
from MCTS import MCTS
from snowman.SnowmanGame import SnowmanGame
from snowman.pytorch.NNet import NNetWrapper as NNet
import numpy as np
from utils import *

HUMAN_PLAY = True
BOARD_LENGTH = 5
LAYER_COUNT = 2
MODEL_FOLDER = './model_s' + str(BOARD_LENGTH) + '_l' + str(LAYER_COUNT) + '/'

class RandomPlayer():
  def __init__(self, game):
    self.game = game

  def play(self, board):
    valids = self.game.getValidMoves(board, 1)
    validCount = np.count_nonzero(valids)
    assert validCount > 0
    probabilities = valids / validCount
    action = np.random.choice(self.game.getActionSize(), p=probabilities)
    return action

class HumanPlayer():
  DIRECTIONS = {"R": 0, "D": 1, "L": 2, "U": 3}

  def __init__(self, game):
    self.game = game

  def play(self, board):
    valids = self.game.getValidMoves(board, 1)
    while True:
      inputMove = input()
      if len(inputMove) == 3:
        try:
          y = ord(inputMove[0].upper()) - ord("A")
          x = int(inputMove[1]) - 1
          d = HumanPlayer.DIRECTIONS[inputMove[2].upper()]
          action = (y * self.game.boardLength + x) * 4 + d
          if valids[action]:
            break
        except ValueError:
          'Invalid move'
      print('Invalid move')
    return action

game = SnowmanGame(BOARD_LENGTH, LAYER_COUNT)

if HUMAN_PLAY:
  player1 = HumanPlayer(game).play
else:
  player1 = RandomPlayer(game).play

nnet = NNet(game)
nnet.load_checkpoint(MODEL_FOLDER, 'best.pth.tar')
args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts = MCTS(game, nnet, args)
player2 = (lambda x: np.argmax(mcts.getActionProb(x, temp=0)))

arena = Arena.Arena(player1, player2, game, display=SnowmanGame.display)

print(arena.playGames(2, verbose=True))
