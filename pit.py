import Arena
from snowman.SnowmanGame import SnowmanGame
import numpy as np
from utils import *

class RandomPlayer():
  def __init__(self, game):
    self.game = game

  def play(self, board):
    valids = self.game.getValidMoves(board, 1)
    validCount = np.count_nonzero(valids)
    if validCount == 0:
      valids = self.game.getValidMoves(board, -1)
    assert validCount > 0
    probabilities = valids / validCount
    action = np.random.choice(self.game.getActionSize(), p=probabilities)
    return action

game = SnowmanGame(6)
player = RandomPlayer(game).play
arena = Arena.Arena(player, player, game, display=SnowmanGame.display)

print(arena.playGames(2, verbose=True))
