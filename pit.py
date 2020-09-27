import Arena
from MCTS import MCTS
from snowman.SnowmanGame import SnowmanGame
from snowman.pytorch.NNet import NNetWrapper as NNet
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

player1 = RandomPlayer(game).play

nnet = NNet(game)
nnet.load_checkpoint('./temp/','best.pth.tar')
args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts = MCTS(game, nnet, args)
player2 = (lambda x: np.argmax(mcts.getActionProb(x, temp=0)))

arena = Arena.Arena(player1, player2, game, display=SnowmanGame.display)

print(arena.playGames(2, verbose=True))
