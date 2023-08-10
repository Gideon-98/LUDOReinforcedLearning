from ludopy.player import Player
from enum import Enum
import numpy as np
from QLearning.actionTable import ActionTable

class State(Enum):
    Home=0 #state in which the token is home, which also is the starting position.
    Safe=1 #state in which the token has reached goal.
    Unsafe=2 # state where the token is safe but not at home or goal.
    Danger=3 #state in which the token is in danger of being knocked home before the next turn.

class Action(Enum):
    #Home
    H_MoveOut=0 # Moving token out of start
    H_Normal=1 # Moving eyes of dice
    H_Goal=2 # Move into goal position
    H_Star=3 #Move on star (jump forward)
    H_Globe=4 # Move to a save point
    H_Protect=5 #Move to same token as yourself -> save
    H_Kill=6 # Kill another player
    H_Die=7 #Move to a field where opponent has 2 or more pieces
    H_GoalZone=8 #Move into goal zone
    H_Nothing=9 #nothing possible

    #Save
    S_MoveOut=10 # Moving token out of start
    S_Normal=11 # Moving eyes of dice
    S_Goal=12 # Move into goal position
    S_Star=13 #Move on star (jump forward)
    S_Globe=14 # Move to a save point
    S_Protect=15 #Move to same token as yourself -> save
    S_Kill=16 # Kill another player
    S_Die=17 #Move to a field where opponent has 2 or more pieces
    S_GoalZone=18 #Move into goal zone
    S_Nothing=19 #nothing possible

    #Unsave
    U_MoveOut=20 # Moving token out of start
    U_Normal=21 # Moving eyes of dice
    U_Goal=22 # Move into goal position
    U_Star=23 #Move on star (jump forward)
    U_Globe=24 # Move to a save point
    U_Protect=25 #Move to same token as yourself -> save
    U_Kill=26 # Kill another player
    U_Die=27 #Move to a field where opponent has 2 or more pieces
    U_GoalZone=28 #Move into goal zone
    U_Nothing=29 #nothing possible

    #Danger
    D_MoveOut=30 # Moving token out of start
    D_Normal=31 # Moving eyes of dice
    D_Goal=32 # Move into goal position
    D_Star=33 #Move on star (jump forward)
    D_Globe=34 # Move to a save point
    D_Protect=35 #Move to same token as yourself -> save
    D_Kill=36 # Kill another player
    D_Die=37 #Move to a field where opponent has 2 or more pieces
    D_GoalZone=38 #Move into goal zone
    D_Nothing=39 #nothing possible


class StateSpace():
    _quarterGameSize=13
    _starPositions=[5,12,18,25,31,38,44,51]
    _globePositionsGlobal=[9,22,35,48]
    _globePositionsLocal=[1]
    _globePositionsEnemyLocal=[]
    _dangerPositionsLocal=[14,27,40]
    LocalPlayerPosition=[Player(), Player(), Player(), Player()]
    GlobalPlayerPosition = [Player(), Player(), Player(), Player()]
    PlayerActionTable=ActionTable(len(State), len(Action))
    _oldVersion=False
    _qLearning=None


    def __init__(self, Debug=0, oldVersion=False):
        super().__init__()
        self._debug=Debug
        self._oldVersion=oldVersion

    def _GlobPos(self,player,piece):
        return self.GlobalPlayerPosition[player].pieces[piece]
    
    def _LocPos(self,player,piece):
        return self.LocalPlayerPosition[player].pieces[piece]

    def UpdateActionTable(self,player,action,piece,value):
        self.PlayerActionTable.UpdateActionTable(action,piece,value)
        # if(self._debug):
        #     print("UpdateActionTable, player: {0}, state: {2}, piece:{3}, action: {1}".format(player,action,self.PlayerActionTable._state, piece))

    def _UpdatePlayerPositions(self, players):
        self.LocalPlayerPosition=players
        indexPlayer = 0
        for player in players:
            indexPiece=0
            for piece in player.pieces:
                if(piece==0):
                    self.GlobalPlayerPosition[indexPlayer].pieces[indexPiece]=0
                elif(piece==59):
                    self.GlobalPlayerPosition[indexPlayer].pieces[indexPiece]=59
                else:
                    self.GlobalPlayerPosition[indexPlayer].pieces[indexPiece]=(piece+(self._quarterGameSize*indexPlayer))%52
                indexPiece = indexPiece+1
            indexPlayer = indexPlayer+1

    def _GetGlobalPosition(self, playerIdx, localPosition):
        if(localPosition==0):
            return 0
        elif(localPosition==59):
            return 59
        else: 
            return (localPosition+(self._quarterGameSize*playerIdx))%52

    def _CheckIfPieceIsSafe(self,player,piece):
        localPos = self._LocPos(player,piece)
        globalPos = self._GlobPos(player,piece)
        isProtected = len(np.where(self.LocalPlayerPosition[player].pieces==localPos)[0])>1
        return self._CheckIfPieceIsSafeAtLoc(localPos, globalPos, isProtected)


    def _CheckIfPieceIsSafeAtLoc(self,localPos, globalPos, isProtected):
        if(globalPos in self._globePositionsGlobal or localPos in self._globePositionsLocal):
            return True
        #check if piece is in goal zone
        if(localPos>=53):
            return True
        if(localPos!=0 and localPos!=59 and isProtected):
            return True
        return False

    def _CheckIfPieceIsInDangerAtLoc(self, localPos, globalPos):
        if(localPos>53 or localPos ==1):
            return False

        if(globalPos in self._globePositionsGlobal):
            return False
        
        if(localPos in self._dangerPositionsLocal):
            return True
        dangerPositions = np.empty
        
        for i in range(1,6):
            dangerPositions = np.append(dangerPositions,np.add(self.enemyList,i))
        if(globalPos in dangerPositions):
            return True
        return False


    def _CheckIfPieceIsInDanger(self,player,piece):
        localPos = self._LocPos(player,piece)
        globalPos = self._GlobPos(player,piece)
        return self._CheckIfPieceIsInDangerAtLoc(localPos, globalPos)
        
        # for i in range(1,6):
        #     if pos-i>=1:
        #         positionsOfInterest.append(pos-i)
        #     else:
        #         positionsOfInterest.append((self._quarterGameSize*4)-(pos-i))

        # for i in range(len(self.GlobalPlayerPosition)):
        #     if i==player:
        #         continue
        #     for posIdx in range(len(self.GlobalPlayerPosition[i].pieces)):
        #         pPos = self._GlobPos(i,posIdx)
        #         if(pPos in positionsOfInterest):
        #             return True
        # return False

    #Return: (killList,dieList)
    def _GetEnemyList(self,player):
        killList = []
        dieList = []
        self.enemyList=[]
        for enemyPlayerIndex in range(len(self.GlobalPlayerPosition)):
            for enemyPieceIndex in range(len(self.GlobalPlayerPosition[enemyPlayerIndex].pieces)):
                enemyPos = self._GlobPos(enemyPlayerIndex, enemyPieceIndex)
                enemyLocPos = self._LocPos(enemyPlayerIndex, enemyPieceIndex)
                if(enemyLocPos in player.pieces):
                    continue
                if(enemyPos>53 or enemyPos==0):
                    continue
                if(enemyPos in self._globePositionsGlobal or enemyLocPos in self._globePositionsLocal):
                    dieList.append(enemyPos)
                    continue
                if(enemyPos in killList):
                    dieList.append(enemyPos)
                    killList.remove(enemyPos)
                else:
                    killList.append(enemyPos)
        enemyList =[]
        enemyList.extend(killList)
        enemyList.extend(dieList)
        return (killList,dieList,enemyList)

    def _GetTargetPlayerState(self,player, piece, dice):
        if(self._oldVersion==True):
            return State.Home
        localPos = self._LocPos(player,piece)+dice
        globalPos = self._GlobPos(player,piece)+dice
        isProtected = len(np.where(self.LocalPlayerPosition[player].pieces==localPos)[0])>=1
        if(self._LocPos(player,piece)==0):
            return State.Home
        if(self._CheckIfPieceIsSafeAtLoc(localPos, globalPos, isProtected)):
            return State.Safe 
        if(self._CheckIfPieceIsInDangerAtLoc(localPos, globalPos)):
            return State.Danger
        return State.Unsafe


    def _SetPlayerState(self, player, piece):
        if(self._LocPos(player,piece)==0):
            self.PlayerActionTable.SetState(State.Home)
        elif(self._CheckIfPieceIsSafe(player,piece)):
            self.PlayerActionTable.SetState(State.Safe)
        elif(self._CheckIfPieceIsInDanger(player,piece)):
            self.PlayerActionTable.SetState(State.Danger)
        else:
            self.PlayerActionTable.SetState(State.Unsafe)

    def _UpdateMoveOutAction(self,player,piece,dice):
        if(self._LocPos(player,piece)==0 and dice==6) :
            nextState = self._GetTargetPlayerState(player, piece, dice).value
            if(self._GetGlobalPosition(player, 1) in self.enemyList):
                self.UpdateActionTable(player, Action(Action.H_Kill.value+nextState*10), piece,1)    
            else:
                self.UpdateActionTable(player, Action(Action.H_MoveOut.value+nextState*10), piece,1)
            # self.UpdateActionTable(player, Action.H_MoveOut, piece,1, dice)
            return True
        return False
        
    def _UpdateNormalAction(self,player,piece,dice):
        if(self._LocPos(player,piece)==0):
            return False
        if(self._LocPos(player,piece)+dice<=59):
            nextState = self._GetTargetPlayerState(player, piece, dice).value
            self.UpdateActionTable(player, Action(Action.H_Normal.value+nextState*10),piece,1)
            return True
    
    def _UpdateGoalAction(self,player,piece,dice):
        localTargetPos = self._LocPos(player,piece)+dice
        if(localTargetPos==59):
            nextState = self._GetTargetPlayerState(player, piece, dice).value
            self.UpdateActionTable(player, Action(Action.H_Goal.value+nextState*10), piece,1)
            return True
        return False

    def _UpdateStarAction(self,player,piece,dice):
        if(self._LocPos(player, piece)==0):
            return False
        if(self._LocPos(player,piece)+dice) in self._starPositions:
            nextState = self._GetTargetPlayerState(player, piece, dice).value
            self.UpdateActionTable(player, Action(Action.H_Star.value+nextState*10), piece,1)
            # self.UpdateActionTable(player, Action.Star,piece,1, dice)
            return True
        return False
    
    def _UpdateGlobeAction(self,player,piece,dice):
        if(self._LocPos(player, piece)==0):
            return False
        if(self._GlobPos(player,piece)+dice) in self._globePositionsGlobal:
            nextState = self._GetTargetPlayerState(player, piece, dice).value
            self.UpdateActionTable(player, Action(Action.H_Globe.value+nextState*10), piece,1)
            # self.UpdateActionTable(player, Action.Globe,piece,1, dice)
            return True
        return False

    def _UpdateProtectAction(self,player, piece,dice):
        if(self._LocPos(player, piece)==0):
            return False
        targetPos = self._LocPos(player,piece)+dice
        if(targetPos>53):
            return False
        for i in range(len(self.LocalPlayerPosition)):
            if(i==piece):
                continue
            if(targetPos == self._LocPos(player,i)):
                nextState = self._GetTargetPlayerState(player, piece, dice).value
                self.UpdateActionTable(player, Action(Action.H_Protect.value+nextState*10), piece,1)
                # self.UpdateActionTable(player, Action.Protect,piece,1, dice)
                return True
        return False

    def _UpdateKillAction(self,player,piece,dice,killList):
        if(self._LocPos(player, piece)==0):
            localTargetPos=1
        else:
            localTargetPos = self._LocPos(player,piece)+dice
        if(localTargetPos>53):
            return False
        
        targetPos = self._GlobPos(player,piece)+dice
        if(targetPos in killList and 
            targetPos not in self._globePositionsGlobal and 
            localTargetPos not in self._globePositionsLocal and
            localTargetPos not in self._dangerPositionsLocal):
            nextState = self._GetTargetPlayerState(player, piece, dice).value
            self.UpdateActionTable(player, Action(Action.H_Kill.value+nextState*10), piece,1)
            # self.UpdateActionTable(player,Action.Kill,piece,1)
            return True
        return False

    def _UpdateDieAction(self,player,piece,dice,dieList):
        if(self._LocPos(player, piece)==0):
            return False
        localTargetPos = self._LocPos(player,piece)+dice
        if(localTargetPos>53):
            return False

        targetPos = self._GlobPos(player,piece)+dice
        if(targetPos in dieList):
            nextState = self._GetTargetPlayerState(player, piece, dice).value
            self.UpdateActionTable(player, Action(Action.H_Die.value+nextState*10), piece,1)
            # self.UpdateActionTable(player, Action.Die,piece,1, dice)
            return True
        return False

    def _UpdateGoalZone(self,player,piece,dice):
        if(self._LocPos(player, piece)==0):
            return False
        localTargetPos = self._LocPos(player,piece)+dice
        if(localTargetPos>53 and localTargetPos<59):
            nextState = self._GetTargetPlayerState(player, piece, dice).value
            self.UpdateActionTable(player, Action(Action.H_GoalZone.value+nextState*10), piece,1)
            # self.UpdateActionTable(player,Action.GoalZone,piece,1, dice)
            return True
        return False

    def _UpdateNotingAction(self,player,piece):
        if(self._LocPos(player, piece)==0):
            return False
        pieceActionTable = self.PlayerActionTable._actionTable[piece]
        if(len(pieceActionTable[np.logical_not(np.isnan(pieceActionTable))])==0):
            self.UpdateActionTable(player,Action.H_Nothing, piece,1)
            return True
        return False

    def GetPossibleActions(self, players, currentPlayer, piecesToMove):
        self._UpdatePlayerPositions(players)
        self.PlayerActionTable.Reset()
        player = players[currentPlayer]
        (killList,dieList,enemyList) = self._GetEnemyList(player)
        self.enemyList=enemyList
        for piece in piecesToMove:
            for dice in range(1,6):
                self._SetPlayerState(currentPlayer, piece)
                # self.CheckGoalZone(currentPlayer,piece)
                self._UpdateMoveOutAction(currentPlayer,piece,dice)
                self._UpdateGoalAction(currentPlayer,piece,dice)
                self._UpdateStarAction(currentPlayer,piece,dice)
                self._UpdateGlobeAction(currentPlayer,piece,dice)
                self._UpdateProtectAction(currentPlayer,piece,dice)
                self._UpdateKillAction(currentPlayer,piece,dice,killList)
                self._UpdateDieAction(currentPlayer,piece,dice,dieList)
                self._UpdateGoalZone(currentPlayer,piece,dice)
                self._UpdateNormalAction(currentPlayer,piece,dice)
                self._UpdateNotingAction(currentPlayer,piece)

    def CheckGoalZone(self, player, piece, dice):
        localPos = self._LocPos(player,piece)
        localTargetPos = localPos+dice
        if(localTargetPos<53):
            return False
        if(localPos>=53):
            self.PlayerActionTable.SetState(State.Safe)
        if(localTargetPos>59):
            if(self._oldVersion):
                self.UpdateActionTable(player, Action(Action.H_Nothing), piece,1)
            else:
                self.UpdateActionTable(player, Action(Action.S_Nothing), piece,1)
            return True
        if(localTargetPos==59):
            if(self._oldVersion):
                self.UpdateActionTable(player, Action(Action.H_Goal), piece,1)
            else:
                self.UpdateActionTable(player, Action(Action.S_Goal), piece,1)
            return True
        
        if(self._oldVersion):
            self.UpdateActionTable(player, Action(Action.H_GoalZone), piece,1)
        else:
            self.UpdateActionTable(player, Action(Action.S_GoalZone), piece,1)
        return True


    def Update(self, players,currentPlayer, piecesToMove, dice):
        self._UpdatePlayerPositions(players)
        self.PlayerActionTable.Reset()
        player = players[currentPlayer]
        (killList,dieList,enemyList) = self._GetEnemyList(player)
        self.enemyList=enemyList
        for piece in piecesToMove:
            self._SetPlayerState(currentPlayer, piece)
            if(self._UpdateMoveOutAction(currentPlayer,piece,dice)):
                continue
            if(self.CheckGoalZone(currentPlayer, piece, dice)):
                continue
            # if(self._UpdateGoalAction(currentPlayer,piece,dice)):
            #     continue
            # if(self._UpdateGoalZone(currentPlayer,piece,dice)):
            #     continue
            if(self._UpdateDieAction(currentPlayer,piece,dice,dieList)):
                continue
            if(self._UpdateStarAction(currentPlayer,piece,dice)):
                continue
            if(self._UpdateGlobeAction(currentPlayer,piece,dice)):
                continue
            if(self._UpdateProtectAction(currentPlayer,piece,dice)):
                continue
            if(self._UpdateKillAction(currentPlayer,piece,dice,killList)):
                continue
            if(self._UpdateNormalAction(currentPlayer,piece,dice)):
                continue
            if(self._UpdateNotingAction(currentPlayer,piece)):
                continue
