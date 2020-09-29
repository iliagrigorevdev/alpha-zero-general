import sys
sys.path.append('..')
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class SnowmanNNet(nn.Module):
  def __init__(self, game, args):
    # game params
    self.plane_count, self.board_x, self.board_y = game.getBoardSize()
    self.action_size = game.getActionSize()
    self.args = args

    super(SnowmanNNet, self).__init__()
    self.conv1 = nn.Conv2d(self.plane_count, args.num_conv_channels, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(args.num_conv_channels, args.num_conv_channels, 3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(args.num_conv_channels, args.num_conv_channels, 3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(args.num_conv_channels, args.num_conv_channels, 3, stride=1, padding=1)

    self.bn1 = nn.BatchNorm2d(args.num_conv_channels)
    self.bn2 = nn.BatchNorm2d(args.num_conv_channels)
    self.bn3 = nn.BatchNorm2d(args.num_conv_channels)
    self.bn4 = nn.BatchNorm2d(args.num_conv_channels)

    self.fc1 = nn.Linear(args.num_conv_channels * self.board_x * self.board_y, args.num_lin_channels)
    self.fc_bn1 = nn.BatchNorm1d(args.num_lin_channels)

    self.fc2 = nn.Linear(args.num_lin_channels, args.num_lin_channels)
    self.fc_bn2 = nn.BatchNorm1d(args.num_lin_channels)

    self.fc3 = nn.Linear(args.num_lin_channels, self.action_size)

    self.fc4 = nn.Linear(args.num_lin_channels, 1)

  def forward(self, s):
    s = s.view(-1, self.plane_count, self.board_x, self.board_y)
    s = F.relu(self.bn1(self.conv1(s)))
    s = F.relu(self.bn2(self.conv2(s)))
    s = F.relu(self.bn3(self.conv3(s)))
    s = F.relu(self.bn4(self.conv4(s)))
    s = s.view(-1, self.args.num_conv_channels * self.board_x * self.board_y)

    s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
    s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)

    pi = self.fc3(s)
    v = self.fc4(s)

    return F.log_softmax(pi, dim=1), torch.tanh(v)
