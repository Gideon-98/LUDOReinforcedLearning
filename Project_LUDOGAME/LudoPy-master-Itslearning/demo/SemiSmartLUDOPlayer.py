from QLearning.stateSpace import StateSpace

class SemiSmartLUDOPlayer(StateSpace):
    _myPlayerIdx = 0
    _lastMove=None
    _piecesInGame = 0


    def __init__(self, myPlayerIdx):
        super().__init__()
        self._myPlayerIdx = myPlayerIdx

    def update(self,players,piecesToMove, dice):
        super().Update(players, self._myPlayerIdx, piecesToMove, dice)
        return self.PlayerActionTable.GetActionTable()