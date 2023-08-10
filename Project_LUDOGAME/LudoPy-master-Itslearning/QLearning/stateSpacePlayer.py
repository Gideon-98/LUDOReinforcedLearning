from QLearning.QTable import Rewards
from QLearning.stateSpace import StateSpace, State, Action
import numpy as np
class StateSpacePlayer(StateSpace):
    _myPlayerIdx = -1
    _debug=False
    qLearning = None
    _state=None
    _action=None

    def __init__(self, myPlayerIdx, oldVersion=False, learning=True, debug=False, friendlyName="", gamma=0.3, learningRate=0.2):
        super().__init__(oldVersion=oldVersion)
        if(oldVersion):
            self.qLearning = Rewards(len(State), len(Action), learning=learning, name="10States_"+friendlyName+"_", gamma=gamma, learningRate=learningRate)
        else:
            self.qLearning = Rewards(len(State), len(Action), learning=learning, name="40States"+friendlyName+"_", gamma=gamma, learningRate=learningRate)
        self._debug=debug
        self._myPlayerIdx = myPlayerIdx

    def update(self, players, piecesToMove, dice):
        super().Update(players, self._myPlayerIdx, piecesToMove, dice)
        actionTable = self.PlayerActionTable.GetActionTable()
        state,action = self.qLearning.ChooseNextAction(self._myPlayerIdx, actionTable)
        piecesToMove = self.PlayerActionTable.GetPieceToMove(state, action)
        if(self._debug):
            print ("Dice: {3}: {0}, {1}".format(State(state), Action(action), piecesToMove, dice))
        self._state=state
        self._action=action

        return piecesToMove
    
    def reward(self, players, piecesToMove):
        super().GetPossibleActions(players, self._myPlayerIdx, piecesToMove)
        newActionTable = np.nan_to_num(self.PlayerActionTable.GetActionTable(), nan=0.0)
        self.qLearning.Reward(self._state, newActionTable, self._action)
    
