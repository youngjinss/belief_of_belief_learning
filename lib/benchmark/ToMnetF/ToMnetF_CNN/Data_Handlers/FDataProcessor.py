import numpy as np

"""
@author: Chuang, Yun-Shiuan; Edwinn
"""
"""
@modified by: Filip Borowiak
"""

"""
The data stored like: 1x12x12x10. 1 - Time Step, 12x12 - Map Resolution, 10 - Depth (1 walls, 1 player, 4 goals, 4 actions)
"""


class DataProcessor:

    def __init__(self, ts, w, h, d):
        self.MAX_TRAJECTORY_SIZE = ts # 20-50
        self.MAZE_WIDTH = w # 13
        self.MAZE_HEIGHT = h # 13
        self.MAZE_DEPTH = d # 11 (1player + 1wall + 4goals + 5 actions = 11)

    def zeroPadding(self, max_elements, all_games):

            # all_games = {
            # "traj_history": data_trajectories,           # Trajectories until state
            # "traj_history_zp": traj_history_zp           # Trajectory with Zero Padding
            # "current_state_history": data_current_state, # (1walls + 1player + 4goals)
            # "actions_history": data_actions,             # actions
            # "labels_history": data_labels                # goals
            #}
        uniform_shape = (1, self.MAZE_DEPTH, self.MAZE_WIDTH, self.MAZE_HEIGHT, max_elements)

        zero_padded_trajectories = []   # ndarray, not list
        unfolded_current_states = []
        unfolded_action_history = []
        unfolded_goal_history = []
        all_trajectories = all_games["traj_history"]
        all_current_states = all_games["current_state_history"]
        all_actions = all_games["actions_history"]
        all_labels = all_games['labels_history']
        N_all_games = len(all_trajectories)

        # Go one by one game
        # Where each game consist of many trajectories
        tracker_var = 0
        for i in range(N_all_games):

            traj = all_trajectories[i]
            cur = all_current_states[i]
            act = all_actions[i]
            goal = all_labels[i]
            N_traj = len(traj) # traj.shape[0]      # Number of trajectories in current game

            for j in range(N_traj):

                ### Init single piece of data from a game -> j = one game
                current_trajectory = traj[j]
                current_state = cur[j]
                current_action = act[j]
                current_goal = goal[j]

                ### Trajectory
                zero_pad_trajectory = np.zeros(shape=uniform_shape)
                Nt = current_trajectory.shape[-1]  # Number of real steps in the trajectory

                # Save game in a bigger array so the rest is fiiled with zeros
                if Nt > max_elements:
                    zero_pad_trajectory[0, ...] = current_trajectory[:, :, :, -max_elements:]
                else:
                    zero_pad_trajectory[0, :, :, :, 0:Nt] = current_trajectory

                zero_padded_trajectories.append(zero_pad_trajectory[0,...])

                ### Current state
                unfolded_current_states.append(current_state)

                ### Action
                unfolded_action_history.append(current_action)

                ### Goal
                unfolded_goal_history.append(current_goal)


            # Keep track on progress
            if i >= int(N_all_games * tracker_var / 100) - 2:
                print('Zero-Padded data ' + str(tracker_var) + '%')
                tracker_var += 5

        zero_padded_trajectories = np.array(zero_padded_trajectories)
        unfolded_current_states = np.array(unfolded_current_states)
        unfolded_action_history = np.array(unfolded_action_history)
        unfolded_goal_history = np.array(unfolded_goal_history)
        
        all_games["traj_history"] = all_trajectories
        all_games["traj_history_zp"] = zero_padded_trajectories
        all_games["current_state_history"] = unfolded_current_states
        all_games["actions_history"] = unfolded_action_history
        all_games["labels_history"] = unfolded_goal_history

        print( f"traj_zp history shape: ")
        print(f"samples: {all_games['traj_history_zp'].shape[0]}")
        print(f"Depth: {all_games['traj_history_zp'].shape[4]}")
        print(f"map size {all_games['traj_history_zp'].shape[2]} x {all_games['traj_history_zp'].shape[3]}")
        print(f"trajectory size: {all_games['traj_history_zp'].shape[1]}")

        print("Zero Padding was applied!")

        return all_games
    