import numpy as np
import random as r
from QLearning.stateSpace import Action
from QLearning.stateSpace import State
import os.path

class Rewards:
    RewardsTable=np.zeros(len(Action))

    QTable=None
    folder="Savegames/"
    tableName = "SaveGame_QTable.txt"
    iterationName = "SaveGame_Iteration.txt"
    iteration = 0
    def __init__(self, states,actions, epsilonGreedy=0.9, gamma=0.3, learningRate=0.2, learning=True, name=""):
        self.tableName = self.folder+name+"SaveGame_QTable.txt"
        self.iterationName = self.folder+name+"SaveGame_Iteration.txt"
        super().__init__()
        self.learning=learning
        if(os.path.isfile(self.tableName)):
            self.QTable = np.loadtxt(self.tableName, delimiter=',')
            print("Load existing q table: {0}".format(name))
            if(not learning):
                epsilonGreedy=1
            if(len(self.QTable)==0):
                self.QTable = np.zeros([states,actions])
            elif (os.path.isfile(self.iterationName)):
                self.iteration = np.loadtxt(self.iterationName, delimiter=',')
        else:
            print("No q table found: {0}".format(name))
            self.QTable = np.zeros([states,actions])
        
        self._epsilonGreedy = epsilonGreedy
        self._gamma = gamma
        self._lr = learningRate

        self.RewardsTable[Action.S_MoveOut.value]=0.25+0.2
        self.RewardsTable[Action.S_Normal.value]=0.01+0.2
        self.RewardsTable[Action.S_Goal.value]=0.8+0.2
        self.RewardsTable[Action.S_Star.value]=0.5+0.2
        self.RewardsTable[Action.S_Globe.value]=0.4+0.2
        self.RewardsTable[Action.S_Protect.value]=0.3+0.2
        self.RewardsTable[Action.S_Kill.value]=0.4+0.2
        self.RewardsTable[Action.S_Die.value]=0+0.2
        self.RewardsTable[Action.S_GoalZone.value]=0.4+0.2

        self.RewardsTable[Action.U_MoveOut.value]=self.RewardsTable[Action.S_MoveOut.value]-0.1
        self.RewardsTable[Action.U_Normal.value]=self.RewardsTable[Action.S_Normal.value]-0.1
        self.RewardsTable[Action.U_Goal.value]=self.RewardsTable[Action.S_Goal.value]-0.1
        self.RewardsTable[Action.U_Star.value]=self.RewardsTable[Action.S_Star.value]-0.1
        self.RewardsTable[Action.U_Globe.value]=self.RewardsTable[Action.S_Globe.value]-0.1
        self.RewardsTable[Action.U_Protect.value]=self.RewardsTable[Action.S_Protect.value]-0.1
        self.RewardsTable[Action.U_Kill.value]=self.RewardsTable[Action.S_Kill.value]-0.1
        self.RewardsTable[Action.U_Die.value]=self.RewardsTable[Action.S_Die.value]-0.1
        self.RewardsTable[Action.U_GoalZone.value]=self.RewardsTable[Action.S_GoalZone.value]-0.1

        self.RewardsTable[Action.D_MoveOut.value]=self.RewardsTable[Action.S_MoveOut.value]-0.2
        self.RewardsTable[Action.D_Normal.value]=self.RewardsTable[Action.S_Normal.value]-0.2
        self.RewardsTable[Action.D_Goal.value]=self.RewardsTable[Action.S_Goal.value]-0.2
        self.RewardsTable[Action.D_Star.value]=self.RewardsTable[Action.S_Star.value]-0.2
        self.RewardsTable[Action.D_Globe.value]=self.RewardsTable[Action.S_Globe.value]-0.2
        self.RewardsTable[Action.D_Protect.value]=self.RewardsTable[Action.S_Protect.value]-0.2
        self.RewardsTable[Action.D_Kill.value]=self.RewardsTable[Action.S_Kill.value]-0.2
        self.RewardsTable[Action.D_Die.value]=self.RewardsTable[Action.S_Die.value]-0.2
        self.RewardsTable[Action.D_GoalZone.value]=self.RewardsTable[Action.S_GoalZone.value]-0.2

        self.RewardsTable[Action.H_MoveOut.value]=self.RewardsTable[Action.S_MoveOut.value]
        self.RewardsTable[Action.H_Normal.value]=self.RewardsTable[Action.S_Normal.value]
        self.RewardsTable[Action.H_Goal.value]=self.RewardsTable[Action.S_Goal.value]
        self.RewardsTable[Action.H_Star.value]=self.RewardsTable[Action.S_Star.value]
        self.RewardsTable[Action.H_Globe.value]=self.RewardsTable[Action.S_Globe.value]
        self.RewardsTable[Action.H_Protect.value]=self.RewardsTable[Action.S_Protect.value]
        self.RewardsTable[Action.H_Kill.value]=self.RewardsTable[Action.S_Kill.value]
        self.RewardsTable[Action.H_Die.value]=self.RewardsTable[Action.S_Die.value]
        self.RewardsTable[Action.H_GoalZone.value]=self.RewardsTable[Action.S_GoalZone.value]
        # self.RewardsTable[Action.Nothing.value]=0

    def GetStateActionOfArray(self,value,array):
        if(np.isnan(value)):
            return (-1,-1)
        idx = np.where(array==value)
        rIdx = r.randint(0,len(idx[0])-1)
        state =idx[0][rIdx]
        action =idx[1][rIdx]
        return (state,action)


    def ChooseNextAction(self,player,actionTable):
        try:
            qTableOptions = np.multiply(self.QTable,actionTable)
        except Exception as e:
            print(e)
        if(r.uniform(0,1)>self._epsilonGreedy):
            self.iteration = self.iteration+1
            nz = actionTable[np.logical_not(np.isnan(actionTable))]
            if(len(nz)<1):
                raise RuntimeError("array to small...")
            randomValue = nz[r.randint(0,len(nz)-1)]
            state,action= self.GetStateActionOfArray(randomValue,actionTable)
        else:
            maxVal = np.nanmax(qTableOptions)
            if(not np.isnan(maxVal)):
                state,action = self.GetStateActionOfArray(maxVal,qTableOptions)
            else:
                nz = actionTable[np.logical_not(np.isnan(actionTable))]
                if(len(nz)!=1):
                    print("Something is wrong... len nz is: {0}".format(len(nz)))
                    raise RuntimeError("Array has not expected size")
                randomValue = nz[r.randint(0,len(nz)-1)]
                state,action= self.GetStateActionOfArray(randomValue,actionTable)
            # state = np.where(qTableOptions==maxVal)[0][0]
            # action = np.where(qTableOptions==maxVal)[1][0]
        return (state,action)

    def Reward(self,state, newActionTable, action):
        state = int(state)
        action = int(action)
        if(not self.learning):
            return
        # newState = int(newState)
        try:
            oldQ = self.QTable[state, action]
            # print("Todo: rate rewards differnetly dependent on next state...")
            deltaQ = self._lr * (self.RewardsTable[action] + self._gamma * np.max(self.QTable*newActionTable) - self.QTable[state, action])
            # if(action == Action.Die.value or action == Action.Die.value):
                # print("oldState: {1}, newState: {2}, newQ: {0}".format(oldQ+deltaQ, state, newState))
                
            self.QTable[state,action] =  oldQ+deltaQ
            # print("update q table, state: {0}, action:{1}".format(state,action))
        except:
            raise RuntimeError("God damn it")
            print("God damn it")

        
    def __del__(self):
        np.savetxt(self.tableName, self.QTable, delimiter=',',fmt='%1.6f')
        np.savetxt(self.iterationName, [self.iteration], delimiter=',',fmt='%1.3f')
