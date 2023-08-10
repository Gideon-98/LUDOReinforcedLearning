import sys
sys.path.insert(0,"../")
sys.path.insert(1,"../LUDO_QLearning/LUDOpy-QLearn/test/")
import progressbar
from QLearning.stateSpacePlayer import StateSpacePlayer
import ludopy
import csv
5

import matplotlib.pyplot as plt
import numpy as np  

def startTesting(learningRate, discountFactor, iterations, players, oldVersion=False, againstQPlayer=True):
    from Qlearn import Qplayer

    resultName="OV_{4}_{0}_pl{3}_df_{1}_lr{2}_againstQPlayer_{5}.csv".format("finalGameAgainstQPlayerNathan", discountFactor, learningRate, players, oldVersion, againstQPlayer)
    if players not in {2,4}:
        raise ValueError("results: players must 2 or 4")
    nameQTable="OV_{4}_{0}_pl{3}_df_{1}_lr{2}_.csv".format("test", discountFactor, learningRate, 4, oldVersion)
    #houskeeping variables
    player1WinningAvg = []
    player2WinningAvg = []
    lastQTable = []
    qTableChange=[]
    idx = []
    debug=False
    learning=False
    player1Won=0

    
    if(players==4):
        g = ludopy.Game(ghost_players=[])
    else:
        g = ludopy.Game(ghost_players=[1,3])
    aiPlayer1 = StateSpacePlayer(0,
    oldVersion=oldVersion,
    learning=learning,
    debug=debug,
    learningRate=learningRate,
    friendlyName=nameQTable,
    gamma=discountFactor)
    playerNathan = 2
    qPlayer = Qplayer()
    lastQTable = aiPlayer1.qLearning.QTable.copy()


    for i in range(1, iterations+1):
        there_is_a_winner = False
        g.reset()
        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
            there_is_a_winner), player_i = g.get_observation()

            if len(move_pieces):
                if(aiPlayer1._myPlayerIdx == player_i):
                    piece_to_move = aiPlayer1.update(g.players,move_pieces, dice)
                    if(not piece_to_move in move_pieces):
                        boardImg = g.render_environment()
                elif(playerNathan == player_i and againstQPlayer):
                    qPlayer.nextmove(player_i, player_pieces, enemy_pieces, dice, move_pieces)
                    piece_to_move = qPlayer.piece
                else:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1
            _, _, _, _, playerIsAWinner, there_is_a_winner = g.answer_observation(piece_to_move)
            if(aiPlayer1._myPlayerIdx == player_i and piece_to_move!=-1):
                aiPlayer1.reward(g.players, [piece_to_move])

        #let's save some values to do some basic stats afterwards...
        if(g.first_winner_was==aiPlayer1._myPlayerIdx):
            player1WinningAvg.append(1)
        else:
            player1WinningAvg.append(0)
            
        if(g.first_winner_was==playerNathan and againstQPlayer):
            player2WinningAvg.append(1)
        else:
            player2WinningAvg.append(0)

        if(len(lastQTable)>0):
            diff = np.subtract(aiPlayer1.qLearning.QTable, lastQTable)
            qTableChange.append(np.sum(np.abs(diff)))
        lastQTable = aiPlayer1.qLearning.QTable.copy()
        idx.append(i)

    # let's save what we got...
    resultArray = np.array([idx,player1WinningAvg, qTableChange, player2WinningAvg])
    np.savetxt("TestResults_New/"+resultName, resultArray, delimiter=',', fmt='%1.9f', header="idx, player1WinningAvg, qTableChange, player2WinningAvg")


def startLearning(learningRate, discountFactor, iterations, players, oldVersion=False):
    friendlyName="test"
    if players not in {2,4}:
        raise ValueError("results: players must 2 or 4")
    name="OV_{4}_{0}_pl{3}_df_{1}_lr{2}_.csv".format(friendlyName, discountFactor, learningRate, players, oldVersion)
    #houskeeping variables
    player1WinningAvg = []
    lastQTable = []
    qTableChange=[]
    idx = []
    debug=False
    learning=True
    player1Won=0

    
    if(players==4):
        g = ludopy.Game(ghost_players=[])
    else:
        g = ludopy.Game(ghost_players=[1,3])
    aiPlayer1 = StateSpacePlayer(0,
    oldVersion=oldVersion,
    learning=learning,
    debug=debug,
    learningRate=learningRate,
    friendlyName=name,
    gamma=discountFactor)
    lastQTable = aiPlayer1.qLearning.QTable.copy()


    for i in range(1, iterations+1):
        there_is_a_winner = False
        g.reset()
        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
            there_is_a_winner), player_i = g.get_observation()

            if len(move_pieces):
                if(aiPlayer1._myPlayerIdx == player_i):
                    piece_to_move = aiPlayer1.update(g.players,move_pieces, dice)
                    if(not piece_to_move in move_pieces):
                        boardImg = g.render_environment()
                else:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1
            _, _, _, _, playerIsAWinner, there_is_a_winner = g.answer_observation(piece_to_move)
            if(aiPlayer1._myPlayerIdx == player_i and piece_to_move!=-1):
                aiPlayer1.reward(g.players, [piece_to_move])

        #let's save some values to do some basic stats afterwards...
        if(g.first_winner_was==aiPlayer1._myPlayerIdx):
            player1WinningAvg.append(1)
            player1Won = player1Won+1
        else:
            player1WinningAvg.append(0)

        if(len(lastQTable)>0):
            diff = np.subtract(aiPlayer1.qLearning.QTable, lastQTable)
            qTableChange.append(np.sum(np.abs(diff)))
        lastQTable = aiPlayer1.qLearning.QTable.copy()
        idx.append(i)

    # let's save what we got...
    resultArray = np.array([idx,player1WinningAvg, qTableChange])
    np.savetxt("TestResults/"+name, resultArray, delimiter=',', fmt='%1.9f', header="idx, player1WinningAvg, qTableChange")
    # with open(name, 'w') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(['idx', 'player1WinningRate', 'qTableChange'])
    #     writer.writerows(enumerate(resultArray))

  

def randwalk():
    import ludopy
    from Qlearn import Qplayer
    from PIL import Image as pilImg
    from QLearning.QTable import Rewards
    from QLearning.stateSpace import Action
    from QLearning.stateSpace import State
    from QLearning.stateSpacePlayer import StateSpacePlayer
    from QLearning.stateSpace import StateSpace
    import numpy as np


    debug = False
    learning=False
    playerNathan = 2
    # g = ludopy.Game(ghost_players=[1,3])
    g = ludopy.Game(ghost_players=[])
    player1Won=0
    player1Avg=0
    player1MovAvg = []
    player2Won=0
    player1WinningAvg = []
    lastQTable = []
    qTableChange=[]
    trainingGames = 500
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    ax1=fig1.add_subplot()
    ax2 = fig2.add_subplot()
    ax1.axis([0, trainingGames, 0, 1])
    ax2.axis([0, trainingGames, 0, 1])
    plt.ion()
    aiPlayer1 = StateSpacePlayer(0,oldVersion=False, learning=learning, debug=debug)
    aiPlayer2 = StateSpacePlayer(2,oldVersion=True, learning=learning, debug=debug)
    player2 = Qplayer()
    for i in progressbar.progressbar(range(1, trainingGames+1)):
        there_is_a_winner = False
        g.reset()
        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
            there_is_a_winner), player_i = g.get_observation()

            if len(move_pieces):
                # if(player_i ==aiPlayer2._myPlayerIdx):
                #     piece_to_move = aiPlayer2.update(g.players,move_pieces, dice)
                if(player_i == playerNathan):
                    player2.nextmove(player_i, player_pieces, enemy_pieces, dice, move_pieces)
                    piece_to_move = player2.piece


                elif(aiPlayer1._myPlayerIdx == player_i):
                    piece_to_move = aiPlayer1.update(g.players,move_pieces, dice)
                    if(not piece_to_move in move_pieces):
                        boardImg = g.render_environment()
                        # img = pilImg.fromarray(boardImg)
                        # img.save("test.jpeg")
                        # raise RuntimeError("I fucked it up again, cannot move this piece...")
                

                else:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                
                # print("Dice: {0}".format(dice))
                
                
                
            else:
                piece_to_move = -1
            # try:
            _, _, _, _, playerIsAWinner, there_is_a_winner = g.answer_observation(piece_to_move)
            #let's get the reward...
            if(aiPlayer1._myPlayerIdx == player_i and piece_to_move!=-1):
                aiPlayer1.reward(g.players, [piece_to_move])
            # if(aiPlayer2._myPlayerIdx == player_i and piece_to_move!=-1):
            #     aiPlayer2.reward(g.players, [piece_to_move])
            if(debug):
                boardImg = g.render_environment()
                img = pilImg.fromarray(boardImg)
                img.save("test.jpeg")

                #let's plot the data

            # except Exception as e:
            #     boardImg = g.render_environment()
            #     img = pilImg.fromarray(boardImg)
            #     img.save("test.jpeg")
            #     print("Damn it, not again!!!!!!!!!!!1")
            #     print(e)
            #     # g.save_hist("game_history.npy")
            #     # print("Saving game video")
            #     # g.save_hist_video("game_video.mp4")
            #     return
        if(g.first_winner_was==aiPlayer1._myPlayerIdx):
            player1WinningAvg.append(1)
            player1Won = player1Won+1
            player1Avg = player1Avg + 1
        else:
            player1WinningAvg.append(0)
        if(g.first_winner_was ==playerNathan):
            player2Won = player2Won+1
        # if(i%20==0):
        #     player1Avg = player1Avg                
        #     plt.plot(player1WinningAvg)
        if(there_is_a_winner):
            if(len(lastQTable)>0):
                diff = np.subtract(aiPlayer1.qLearning.QTable, lastQTable)
                qTableChange.append(np.sum(np.abs(diff)))

            # fig1.clf()
            lastQTable = aiPlayer1.qLearning.QTable.copy()
            movingAvg = np.convolve(player1WinningAvg, np.ones(50), 'valid')/50
            player1MovAvg.append(player1Won/i)
            if(i%10==0):
                ax1.plot(movingAvg, 'g')
                ax1.plot(player1MovAvg, 'r')
                ax2.plot(qTableChange, 'r')
            # fig1.show()
                plt.pause(0.1)
                    # fig1.cla()
            # plt.plot(player1WinningAvg, 'b', linestyle="None")
        # print("Success... Saving history to numpy file")

    # g.save_hist("game_history.npy")
        # print("Saving game video")
    # g.save_hist_video("game_video.mp4")
    fig1.savefig("SuccessRate.png")
    fig2.savefig("QTableChange.png")
    print("Success Jan: {0}%".format((player1Won/trainingGames)*100))
    print("Success Nathan: {0}%".format((player2Won/trainingGames)*100))

    return True


def testWithLearningRate(learningRate):
    print("learning rate: {0}".format(learningRate))
    for discountFactor in progressbar.progressbar(range(0,8,1)):
        startLearning(players=2, iterations=500, learningRate=learningRate/10, discountFactor=discountFactor/10)
        startLearning(players=4, iterations=500, learningRate=learningRate/10, discountFactor=discountFactor/10)


def startParallelLearning():
    # Parallel(n_jobs=5)(delayed(testWithLearningRate)(i) for i in [0.4,0.6,0.8,1,1.2,1.4,2,3,4])
    proc=[]
    # proc.append(Process(target = startLearning, args=(0.03, 0.2, 200, 2,True)))
    # proc.append(Process(target = startLearning, args=(0.1, 0.0, 200, 2,True)))
    # proc.append(Process(target = startLearning, args=(0.06, 0.2, 200, 2,True)))
    # proc.append(Process(target = startLearning, args=(0.12, 0.2, 500, 4)))
    proc.append(Process(target = startLearning, args=(0.2, 0.2, 500, 4, False)))
    proc.append(Process(target = startLearning, args=(0.2, 0.2, 500, 2, False)))

    proc.append(Process(target = startLearning, args=(0.2, 0.2, 500, 4, True)))
    proc.append(Process(target = startLearning, args=(0.2, 0.2, 500, 2, True)))
    
    # proc.append(Process(target = startLearning, args=(0.2, 0.2, 500, 4, False, False)))
    # proc.append(Process(target = startLearning, args=(0.2, 0.2, 500, 2, False, False)))
    # proc.append(Process(target = startLearning, args=(0.2, 0.2, 500, 4, False)))

    # proc.append(Process(target = startLearning, args=(0.2, 0.4, 500, 4)))
    for p in proc:
        p.start()
    for p in proc:
        p.join()

def startParallelTesting():
    proc=[]
    # proc.append(Process(target = startTesting, args=(0.03, 0.2, 500, 2,True)))
    # proc.append(Process(target = startTesting, args=(0.1, 0.0, 500, 2,True)))
    # proc.append(Process(target = startTesting, args=(0.06, 0.2, 500, 2,True)))
    # proc.append(Process(target = startTesting, args=(0.12, 0.2, 2000, 4)))
    # proc.append(Process(target = startTesting, args=(0.2, 0.2, 2000, 4)))
    # proc.append(Process(target = startTesting, args=(0.2, 0.4, 2000, 4)))
    
    # 4 players, against nathan
    proc.append(Process(target = startTesting, args=(0.06, 0.2, 2000, 2, False, True)))
    # proc.append(Process(target = startTesting, args=(0.06, 0.2, 200, 4, False, True)))
    # proc.append(Process(target = startTesting, args=(0.06, 0.2, 200, 2, False, True)))

    # proc.append(Process(target = startTesting, args=(0.06, 0.2, 200, 4, False, False)))
    # proc.append(Process(target = startTesting, args=(0.06, 0.2, 200, 2, False, False)))
    # proc.append(Process(target = startTesting, args=(0.2, 0.2, 2000, 2, False, True)))

    # proc.append(Process(target = startTesting, args=(0.2, 0.2, 2000, 4, True, True)))
    # proc.append(Process(target = startTesting, args=(0.2, 0.2, 2000, 2, True, True)))
    
    # proc.append(Process(target = startTesting, args=(0.2, 0.2, 2000, 4, False, False)))
    # proc.append(Process(target = startTesting, args=(0.2, 0.2, 2000, 2, False, False)))
    # proc.append(Process(target = startTesting, args=(0.2, 0.2, 2000, 2, True)))
    # proc.append(Process(target = startTesting, args=(0.2, 0.2, 2000, 2, False)))
    # proc.append(Process(target=startLearning, args=(0.3, 0.2, 20200, 4)))
    # proc.append(Process(target=startLearning, args=(0.1, 0.0, 2000, 4)))
    # proc.append(Process(target=startTesting, args=(0.06, 0.2, 2000, 4)))
    # proc.append(Process(target=startTesting, args=(0.06, 0.2, 2000, 2)))
    for p in proc:
        p.start()
    for p in proc:
        p.join()

if __name__ == '__main__':
    from joblib import Parallel, delayed
    from multiprocessing import Process
    # startParallelLearning()
    startParallelTesting()
    # 
    # p1.start()
    # def startLearning(learningRate, discountFactor, iterations, players
    # proc=[]
    # # proc.append(Process(target=startLearning, args=(0.3, 0.2, 2000, 4)))
    # # proc.append(Process(target=startLearning, args=(0.1, 0.0, 2000, 4)))
    # # proc.append(Process(target=startTesting, args=(0.06, 0.2, 2000, 4)))
    # # proc.append(Process(target=startTesting, args=(0.06, 0.2, 2000, 2)))
    # for p in proc:
    #     p.start()
    # for p in proc:
    #     p.join()
    # print ("done")
    # p2.start()
    # p1.join()
    # p2.join()
    
    # startLearning(players=4, iterations=2000, learningRate=0.03, discountFactor=0.2)
    # startLearning(players=4, iterations=2000, learningRate=0.1, discountFactor=0.0)
    # startLearning(players=4, iterations=2000, learningRate=0.06, discountFactor=0.2)
    # for learningRate in progressbar.progressbar(range(0,5,1)):)
        