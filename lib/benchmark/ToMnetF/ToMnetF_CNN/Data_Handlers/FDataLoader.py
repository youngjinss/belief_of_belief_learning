import os
import numpy as np
import re
import matplotlib.pyplot as plt
import torch

"""
@author: Chuang, Yun-Shiuan; Edwinn
"""
"""
@modified by: Filip Borowiak
"""

class DataReader():
    def __init__(self, ts, w, h, d, experiment_no):

        self.EXPERIMENT_NO = experiment_no
        self.MAX_TRAJECTORY_SIZE = ts
        self.MAZE_WIDTH = w # 13
        self.MAZE_HEIGHT = h # 13
        self.MAZE_DEPTH_TRAJECTORY = d # 10 (1 - player, 1 - walls, 4 - objects, 4 - directions)

    def check_ifCollected(self, x):
       
        if x[-2] == 'A' or x[-2] == 'B' or x[-2] == 'C' or x[-2] == 'D': return True

        
    def ReadOneGame(self, filename):
        f = open(filename, "r")
        # lines 1: 15 - map
        map = []
        consumed = []
        trajectory_length = []
        trajectory = {}
        actions = []
        steps = []

        traj = np.empty((self.MAZE_DEPTH_TRAJECTORY, self.MAZE_WIDTH, self.MAZE_HEIGHT, 1))
        
        act  = np.empty(1, dtype=np.int8)
        goal = np.empty(1, dtype=np.int8)

        for idx, x in enumerate(f):
            temp_pos = []
            temp_row = []
            
            if idx >= 2 and idx <= 14:
                for idx_2, c in enumerate(x):
                    if idx_2 >=1 and idx_2 <= 13:
                        if c == '#':
                            temp_row.append(0)
                        elif c == 'A':
                            temp_row.append(2)
                        elif c == 'B':
                            temp_row.append(3)
                        elif c == 'C':
                            temp_row.append(4)
                        elif c == 'D':
                            temp_row.append(5)
                        elif c == '-':
                            temp_row.append(1)
                        elif c == 'O':
                            temp_row.append(10)
                    
                map.append(temp_row)

            if idx == 16:
                consumed.append(x)
            if idx == 17:
                trajectory_length.append(x)

            if idx >= 18:
                
                # [(posX, posY), trajectory move]
                if x[0] == '[' and x[2] == "," and x[5] == ']': #both are single digit
                    temp_pos = (int(x[1]), int(x[4]))
                    temp_traj = int(x[9])
                    
                    actions.append(temp_traj)
                    
                    
                elif x[0] == '[' and x[3] == ',' and x[6] == ']': # first is double digit second single digit
                    temp_pos = (int(x[1]+x[2]), int(x[5]))
                    temp_traj = int(x[10])
                    
                    actions.append(temp_traj)
                   
                elif x[0] == '[' and x[2] == "," and  x[6] == ']':# first is single digit second double digit
                    temp_pos = (int(x[1]), int(x[4]+x[5]))
                    temp_traj = int(x[10])
                    
                    actions.append(temp_traj)
                    
                else: # Both double
                    temp_pos = (int(x[1]+x[2]), int(x[5]+x[6]))
                    temp_traj = int(x[11])
                    
                    actions.append(temp_traj)
                    
                trajectory[temp_pos] = temp_traj

        map = np.array(map) # Map
        consumed = consumed[0][19] # Consumed goal

        # Plane for obstacles - static
        np_obstacles = np.where(map == 0, 1, 0).astype(np.int8)   # if wall then 1 else 0   
        
        # Plane for agent's initial position 
        np_agent = np.where(map == 10 , 1, 0).astype(np.int8)

        #        A  B  C  D
        goals = [2, 3, 4, 5]
        np_targets = np.zeros((4, self.MAZE_WIDTH, self.MAZE_HEIGHT))
        for target, i in zip(goals, range(len(goals))):
            np_targets[i ,:, :,] = np.where(map == target, 1, 0).astype(np.int8)

        
        directions = {"Action:0" : 0,
                      "Action:1" : 0,
                      "Action:2" : 0,
                      "Action:3" : 0
                      }
        
        for idx, (key, val) in enumerate(trajectory.items()):
            posX = key[0]
            posY = key[1]
            temp_trajectory = val
            if idx == 0: # first values are init numpy values - replace
                goal[0] = self.goal_sym_to_num(consumed)
                act[0] = val
            else:

                act = np.append(act, val)
                goal = np.append(goal, self.goal_sym_to_num(consumed))

                np_agent = np.zeros(shape=(self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype = np.int8)
                np_agent[posX, posY] = 1

            # Make tensor traj
            #np_actions = np.zeros((self.MAZE_WIDTH, self.MAZE_HEIGHT, 5), dtype=np.int8)
            np_actions = np.zeros((4, self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.int8)
            np_actions[val, int(posX), int(posY)] = 1 # update trajectory taken
            if val == 0:
                directions['Action:0'] += 1
            elif val == 1:
                directions['Action:1'] += 1
            elif val == 2:
                directions['Action:2'] += 1
            elif val == 3:
                directions['Action:3'] += 1
            

            np_obstacles1 = np.expand_dims(np_obstacles, 0)
            np_agent1 = np.expand_dims(np_agent, 0)
            
            tensor = np.concatenate((np_obstacles1, np_agent1, np_targets, np_actions)) # (1walls + 1player + 4goals + 4actions)
            
            steps.append(tensor) # each step (record) is one decision data
            traj = np.stack(steps, axis=-1) # traj consists of many steps records
                
        #traj = torch.tensor(steps)
        return traj, act, goal, directions

       

        
        #print(f"map size: {map.shape}\n")
        #print(map)
        
        #trajectory_length = trajectory_length[0][19:]

    def LoadAllGames(self, use_percentage, directory):
        # Get names of games
        files = os.listdir(directory)
        r = re.compile(".*.txt")
        files = list(filter(r.match, files))
        Nfiles = len(files)
        Nfraction = int(np.ceil(use_percentage * Nfiles))  # Apply a fraction division
        files = files[:Nfraction]
        print("----")
        print("Saved Games found: ", Nfiles)
        print("Saved Games loaded: ", Nfraction)
        print("Percentage of loaded games: ", use_percentage * 100, "%")
        print("Games names: ", files)

        # Save all trajectories and labels
        trajectories = []  # np.empty([1, self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH_TRAJECTORY])
        actions = []  # np.empty(1)
        labels = []  # np.empty(1)
        """
            How to read Actions:
            0 - UP
            1 - RIGHT
            2 - DOWN
            3 - LEFT
        """
        directions_total = {'Action:0': 0,
                            'Action:1': 0,
                            'Action:2': 0,
                            'Action:3': 0}
        directions_mapping = {'Action Up': 0,
                              'Action Right': 1,
                              'Action Down': 2,
                              'Action Left': 3}
        

        # ------------------------------------------------------------------
        # 1. Load each game one by one
        # ------------------------------------------------------------------
        j = 0  # for tracking progress (%)
        for i, file in enumerate(files):
            

            # Read one game
            traj, act, goal, directions = self.ReadOneGame(filename=os.path.join(directory, file))
            
            directions_total['Action:0'] += directions['Action:0']
            directions_total['Action:1'] += directions['Action:1']
            directions_total['Action:2'] += directions['Action:2']
            directions_total['Action:3'] += directions['Action:3']
                

            # Append a game to data
            trajectories.append(traj)
            actions.append(act)
            labels.append(goal)

            # Keep track on progress
            if i >= int(np.ceil(j * Nfraction / 100)) - 1:
                print('Parsed ' + str(j) + '%')
                j += 10
        print("----")

        print("Augment data. One game creates many training samples!")

        data_trajectories = []
        data_current_state = []
        data_actions = []
        data_labels = []
        j = 0  # for tracking progress (%)

        # Process Game-per-Game
        for i in range(Nfraction):

            # Consider only games with more than 6 moves
            if trajectories[i].shape[0] < 6:
                continue

            # Prepare data from one game
            # The dimensions differ, so only list is applicable (no numpy arrays)
            data_trajectories1, data_current_state1, \
            data_actions1, data_labels1 = self.generateDataFromGame(
                trajectories=trajectories[i],
                actions=actions[i],
                labels=labels[i])

            # Append to a single structure
            data_trajectories.append(data_trajectories1)
            data_current_state.append(data_current_state1)
            data_actions.append(data_actions1)
            data_labels.append(data_labels1)

            # Keep track on progress
            if i >= int(np.ceil(j * Nfraction / 100)) - 1:
                print('Augmented data ' + str(j) + '%')
                j += 10

        print("----")

        # data_trajectories1 shape is ()
        all_games = {
            "traj_history": data_trajectories,
            "current_state_history": data_current_state,
            "actions_history": data_actions,
            "labels_history": data_labels
        }
        
        print(f"Directions count: {directions_total}")
        print(f"Directions mapping: {directions_mapping}")
        names = [key for key, _ in directions_mapping.items()]
        values = [val for _, val in directions_total.items()]
        
        plt.bar(names, values)
        plt.grid(True)
        plt.legend()
        plt.title("Distribution of Actions")
        plt.savefig(f"ToMnetF/Results/Model/Experiment{self.EXPERIMENT_NO}/action_distribution")
        plt.show()
            
        return all_games

    def generateDataFromGame(self, trajectories, actions, labels):

            # Make full data from a game
            data_trajectories = []
            data_current_state = []
            data_actions = []
            data_labels = []

            MIN_ACTIONS = 6
            for i in range(MIN_ACTIONS, trajectories.shape[-1]):
                data_trajectories.append(trajectories[:, :, :, 0:i])     # Trajectory to the state
                data_current_state.append(trajectories[0:6, :, :, i]) # Current state # (1walls + 1player + 4goals)
                data_actions.append(actions[i,...])                 # Next Action
                data_labels.append(labels[i,...])                   # Consumed Goal

            return data_trajectories, data_current_state, data_actions, data_labels




    def goal_sym_to_num(self, goal_sym):
        out = 0
        if goal_sym == "A":
            out = 1
        elif goal_sym == "B":
            out = 2
        elif goal_sym == "C":
            out = 3
        elif goal_sym == "D":
            out = 4
        else:
            raise ValueError("ERROR: wrong goal sym was given!")
        return out


