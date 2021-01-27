import sys
import os
import shutil
import enum
import keras
import mtcs
from reversi import Reversi as game


class Player(enum.Enum):
    P1 = 1
    P2 = 2

    def toStateValue(self):
        return 1 if self == Player.P1 else -1

    def opposite(self):
        return Player.P2 if self == Player.P1 else Player.P1


cudaVisibleDevices = "0"
if sys.platform == "linux":
    os.environ["CUDA_VISIBLE_DEVICES"] = cudaVisibleDevices

gameName = game.getGameName()
modelPath = "./1713215.h5"

model = keras.models.load_model(modelPath)

playerLookup = {
    Player.P1: (p1Model, p1Model and TreeNode(None, 1)),
    Player.P2: (p2Model, p2Model and TreeNode(None, 1))}
