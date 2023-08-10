import os
import numpy as np
from numpy import genfromtxt
import re
import fnmatch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from numpy.lib.utils import source
import pandas as pd
from shutil import copyfile

def go(sourceDir="TestResults", targetDir='', pattern="Test_pl2_df_0.0_lr0.*.*", twoPlayerResults=False):
    pltDpi=900
    dataframes = []
    testResults=[]
    idx=[]
    player1Winning=[]
    qTableChange=[]
    xAxisMax = 2000
    name = []
    axes = plt.gca()
    axes.set_ylim([0,1])
    patternFileName =  pattern.replace('.csv', '').replace('.','_').replace('*','x')
    for filename in os.listdir(sourceDir):
        if(re.match(fnmatch.translate(pattern), filename)):
            if(sourceDir != targetDir):
                copyfile(os.path.join(sourceDir, filename), os.path.join(targetDir, filename))
            testResult = np.genfromtxt(os.path.join(sourceDir, filename), delimiter=',')
            # df = pd.DataFrame({'win': testResult[1], 'qTable':testResult[2]})
            if(twoPlayerResults):
                df = pd.DataFrame({'win': testResult[1], 'qTable':testResult[2], 'winPl2': testResult[3]})
            else:
                df = pd.DataFrame({'win': testResult[1], 'qTable':testResult[2]})
            dataframes.append(df)
            name.append(filename)
            # dataframes.append()
            # testResults.append(testResult)
            # idx.append(testResult[0])
            # player1Winning.append(testResult[1])
            # qTableChange.append(testResult[2])
    if (len(dataframes)==0):
        print("Error, could not find file with pattern: "+pattern)
        return
    
    # for i in range(1, len(dataframes)):
    ax=None
    plt.figure()
    legend = []
    fontP = FontProperties()
    # if(len(dataframes)>10):
        # fontP.set_size('xx-small')

    for i in range(0,len(dataframes)):
        df=dataframes[i]
        df=df.expanding().mean()
        ax = df[('win')].plot(ax=ax)
        # legend.append("Discount factor: {0:.2f}, WR: {1:.3%}".format(float(name[i].split('df_')[1].split('_lr')[0]), df['win'].mean()))
        legend.append("Winning average Q-Player".format(int(name[i].split('pl')[1].split('_')[0])-1))
        if(twoPlayerResults):
            ax = df[('winPl2')].plot(ax=ax)
            legend.append("Winning average Nathan Durocher's player")
        print("WinningRate: {0:.3%},   pattern: {1}".format(df['win'].mean(), name[i]))
    ax.legend(["New approach", "Old approach"], prop=fontP)
    ax.axis([0, xAxisMax, 0, 1])
    # title = "Average winning rate against random player(s)"
    title="Average winning rate against Nathan Durocher's player"
    plt.title(title)
    plt.savefig(os.path.join(targetDir, title.replace(':', '')+patternFileName+".png"), dpi=pltDpi)
    plt.close()
    
    ax=None
    plt.figure()
    legend = []
    title = "Q-table change"
    for i in range(0,len(dataframes)):
        df=dataframes[i]
        df=df.rolling(window=5).mean()
        ax = df[('qTable')].plot(ax=ax)
        legend.append("Learning rate: {0}".format(float(name[i].split('lr')[1].split('_')[0])))
        # legend.append(name[i]+"  Mean: {:.3%}".format(df['qTable'].mean()))
    ax.legend(legend, prop=fontP)
    ax.axis([0, 500, 0, 1])
    plt.title(title)
    plt.savefig(os.path.join(targetDir, title.replace(':', '_')+patternFileName+".png"), dpi=pltDpi)
    plt.close()


    # for i in range(1,len(dataframes)):
    #     dataframes[i].plot(ax=ax)
    # for i in range(0, len(player1Winning)):
    #     mean = []
    #     playerWon=0.0
    #     for k in range(0, len(player1Winning[i])):
    #         if(player1Winning[i][k]==1):
    #             playerWon+=1
    #         mean.append(playerWon/(k+1))
    #     wr = playerWon/len(player1Winning[i])
    #     plt.plot(idx[i], mean, label=name[i]+"  WR: {:.3%}".format(wr))


    # plt.legend()
    # title = "Average winning rate: "
    # plt.title(title+pattern)
    # plt.savefig(os.path.join(dir, "Plots",title.replace(':', '')+patternFileName+".png"))
    # plt.close()


    # for i in range(0, len(qTableChange)):
    #     plt.plot(idx[i], qTableChange[i], label=name[i])
    # title = "Q-Table change: "
    # plt.legend()
    
    # plt.title(title+pattern)
    # plt.savefig(os.path.join(dir, "Plots",title.replace(':', '_')+patternFileName+".png"))
    # plt.close()
    


if __name__ == '__main__':
    # learningRate, discountFactor, iterations, players
    # go(pattern="finalGameAgainstQPlayerNathan_pl2_df_0.2_lr0.03_.csv")
    # go(pattern="OV_2_finalGameAgainstQPlayerNathan_pl2_df_0.2_lr0.06_.csv")
    # go(pattern="OV_2_finalGameAgainstQPlayerNathan_pl2_df_0.0_lr0.1_.csv")
    # go(pattern="OV_2_finalGameAgainstQPlayerNathan_pl2_df_0.2_lr0.03_.csv")
    # go(pattern="finalGameAgainstQPlayerNathan_pl4_df_0.2_lr0.06_.csv")
    # go(pattern="finalGameAgainstQPlayerNathan_pl2_df_0.2_lr0.06_.csv")
    # go(pattern="Test_pl2_df_0.2_lr0.06_.csv")
    # go(pattern="Test_pl4_df_0.2_lr0.*_.csv", sourceDir = "TestResults", targetDir="TestResults/Plots/FinalResults")
    # go(pattern="*pl4*_.csv", sourceDir = "TestResults/Plots/FinalResults/QTableChange", targetDir="TestResults/Plots/FinalResults/QTableChange")
    # go(pattern="*pl4*lr{0}_*".format(0.2), sourceDir = "TestResults/Plots/FinalResults/DiscountFactor", targetDir="TestResults/Plots/FinalResults/DiscountFactor")
    # Plots\FinalResults\QTableChange
    go(pattern="*pl*.csv", sourceDir = "TestResults_new/OldVsNew", targetDir="TestResults_new/OldVsNew", twoPlayerResults=False)
    # go(pattern="OV_False_finalGameAgainstQPlayerNathan_pl2_df_0.2_lr0.06_againstQPlayer_True.csv", sourceDir = "TestResults_new/AgainstNathan", targetDir="TestResults_new/AgainstNathan", twoPlayerResults=True)
    # go(pattern="*pl4*.csv", sourceDir = "TestResults_new/2Vs4Random", targetDir="TestResults_new/2Vs4Random", twoPlayerResults=True)
    # go(pattern="*False*pl4*.csv", sourceDir = "TestResults/Plots/FinalResults/FinalGame", targetDir="TestResults/Plots/FinalResults/FinalGame", twoPlayerResults=True)
    # go(pattern="*True*pl2*.csv", sourceDir = "TestResults/Plots/FinalResults/FinalGame", targetDir="TestResults/Plots/FinalResults/FinalGame", twoPlayerResults=True)
    # go(pattern="*True*pl4*.csv", sourceDir = "TestResults/Plots/FinalResults/FinalGame", targetDir="TestResults/Plots/FinalResults/FinalGame", twoPlayerResults=True)

    # go(pattern="Test_pl4_df_0.2*_lr0.06*_.csv")
    # go(pattern="Test_pl2_df_0.*_lr0.*_.csv")
    # go(pattern="Test_pl4_df_0.*_lr0.*_.csv")
    # for df in range(0,5):
    #     go(pattern="Test_pl2_df_0.{0}_lr0.*.csv".format(df), imgDir="2Player")
    #     go(pattern="Test_pl4_df_0.{0}_lr0.*.csv".format(df), imgDir="4Player")
    # for lr in range(0,5):
    #     go(pattern="Test_pl2_df_0.*_lr0.{0}_.csv".format(lr), imgDir="2Player")
    #     go(pattern="Test_pl4_df_0.*_lr0.{0}_.csv".format(lr), imgDir="4Player")