import os
import sys
import ludopy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
from player import QLearningAgent
import csv

def append_to_csv(file_name, data):
    # Open the CSV file in append mode
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data])  # Pass the item as a single-item list

def calculate_mean(data, start_index, end_index):
    subarray = data[start_index:end_index + 1]
    total = sum(subarray)
    mean = total / len(subarray)
    return mean

def remove_digits(number, digits):
    number_str = str(number)
    new_number_str = number_str[:-digits]  # Remove the last 'digits' characters
    new_number = int(new_number_str)
    return new_number



def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def epsilon_decay(epsilon, decay_rate, episode):
    return epsilon * np.exp(-decay_rate*episode)

def start_teaching_ai_agent(episodes, no_of_players, epsilon, epsilon_decay_rate, lr, gamma):
    
    # Houskeeping variables
    ai_player_winning_avg = []
    epsilon_list = []
    idx = []
    ai_player_won = 0

    # Store data
    win_rate_list = []
    max_expected_return_list = []

    if no_of_players == 4:
        g = ludopy.Game(ghost_players=[])
    elif no_of_players == 3:
        g = ludopy.Game(ghost_players=[1])
    else:
        g = ludopy.Game(ghost_players=[1,3])

    ai_player_1 = QLearningAgent(0, learning_rate=lr, gamma=gamma)

    for episode in range(0, episodes):

        there_is_a_winner = False
        g.reset()
        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,there_is_a_winner), player_i = g.get_observation()

            if len(move_pieces):
                if ai_player_1.ai_player_idx == player_i:
                    piece_to_move = ai_player_1.update(g.players, move_pieces, dice)
                    if not piece_to_move in move_pieces:
                        g.render_environment()
                else:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1
            _, _, _, _, playerIsAWinner, there_is_a_winner = g.answer_observation(piece_to_move)
            
            if episode > 1:
                board = g.render_environment()
                cv2.imshow("Ludo Board", board)
                cv2.waitKey(500)
                
            if ai_player_1.ai_player_idx == player_i and piece_to_move != -1:
                ai_player_1.reward(g.players, [piece_to_move])

        #print (ai_player_1.q_learning.q_table)
        #if episode == 200:
        #    g.save_hist_video("game.mp4")
            
        new_epsilon_after_decay = epsilon_decay(epsilon=epsilon, decay_rate=epsilon_decay_rate,episode=episode)
        epsilon_list.append(new_epsilon_after_decay)
        ai_player_1.q_learning.update_epsilon(new_epsilon_after_decay)


        if g.first_winner_was == ai_player_1.ai_player_idx:
            ai_player_winning_avg.append(1)
            ai_player_won = ai_player_won + 1
        else:
            ai_player_winning_avg.append(0)

        idx.append(episode)

        # Print some results
        win_rate = ai_player_won / len(ai_player_winning_avg)
        win_rate_percentage = win_rate * 100
        win_rate_list.append(win_rate_percentage)

        if episode % 100 == 0:
            print("Episode: ", episode)
            print(f"Win rate: {np.round(win_rate_percentage,1)}%")
    
        max_expected_return_list.append(ai_player_1.q_learning.max_expected_reward)
        ai_player_1.q_learning.max_expected_reward = 0


    # Moving averages
    window_size = 20
    cumsum_vec = np.cumsum(np.insert(win_rate_list, 0, 0)) 
    win_rate_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

    cumsum_vec = np.cumsum(np.insert(max_expected_return_list, 0, 0)) 
    max_expected_return_list_ma = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size


    
    moving_average_list = [0] * window_size
    win_rate_ma = moving_average_list + win_rate_ma.tolist()
    max_expected_return_list_ma = moving_average_list + max_expected_return_list_ma.tolist()
    
    
    return win_rate_list, win_rate_ma, epsilon_list, max_expected_return_list, max_expected_return_list_ma

# FINAL AGENT PLAYING AGAINST 1,2 and 3 RANDOM PLAYERS

learning_rate = 0.2 #0.2
gamma = 0.5
epsilon = 0.9
epsilon_decay_rate = 0.09
episodes = 1000
opponents = 3

# Start teaching the agent
win_rate_list, win_rate_ma, epsilon_list, max_expected_return_list, max_expected_return_list_ma = start_teaching_ai_agent(episodes, opponents, epsilon, epsilon_decay_rate, learning_rate, gamma)

start_index = 99
end_index = 999
avg_winrate = round(calculate_mean(win_rate_list, start_index, end_index),2)


# Plot win rates against opponents
fig, axs = plt.subplots(1)
axs.set_title("Win Rate")
axs.set_xlabel('Episodes')
axs.set_ylabel('Win Rate %')
axs.axis([0.0,1000.0, 0.0,100.0])
axs.set_autoscale_on(False)

major_ticks = np.arange(0, 1001, 100)
minor_ticks = np.arange(0, 1001, 50)
major_ticks_y = np.arange(0, 101, 20)
minor_ticks_y = np.arange(0, 101, 5)

axs.set_xticks(major_ticks)
axs.set_xticks(minor_ticks, minor=True)
axs.set_yticks(major_ticks_y)
axs.set_yticks(minor_ticks_y, minor=True)
# And a corresponding grid
axs.grid(which='both')
axs.axvline(x = 100, color = 'b', linestyle = '--')
axs.axhline(y = avg_winrate, color = 'g')
axs.plot(win_rate_list, color = 'tab:red')
axs.legend(['exploration',avg_winrate, opponents])
#plt.savefig('Killer_winrate_3.png')

# Plot epsilon decay
fig, axs = plt.subplots(1)
axs.set_title("Epilson Decay")
axs.set_xlabel('Episodes')
axs.set_ylabel('Epsilon')
axs.plot(epsilon_list, color='tab:red')
axs.legend(['Epsilon Decay'])

# Append the data to the CSV file
#append_to_csv('data_killer_3.csv', avg_winrate)

waitkey = cv2.waitKey(0)

plt.show()


data = {'0.1':59.53, 
        '0.2':58.55,
        '0.3':58.97,
        '0.4':58.44,
        '0.5':58.06,
        '0.6':58.48,
        '0.7':59.02,
        '0.8':58.59,
        '0.9':57.86,
        '1.0':57.17}
courses = list(data.keys())
values = list(data.values())

  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon',
        width = 0.4)
plt.grid(True, color = "grey", linewidth = "0.5",axis = 'y')
 
plt.xlabel("Learning Ratios")
plt.ylabel("Winrates")
plt.title("Go-To-Stars Strategy on different Learning Ratios vs 3 random players")
plt.show()