from Data_Handlers import FDataLoader as DL
from ToMnetEngine.ToMnetF import ToMnet

import torch
import numpy as np
import GameSequenceDrawer as gsd

"""
@Author Filip Borowiak
"""


simple_map = map
W=13
H=13

for i in range(100):
    model = ToMnet(Batch=16, ResidualBlocks=5,
                    N_echar=8, out_channels=32,
                    Max_trajectory_size=10, Width=13,
                        Height=13, Depth=10)

    model.load_state_dict(torch.load("ToMnetF/Results/Model/Experiment1/ToMnetF.pt"))
    #print("model loaded")

    # get new MDP
    dataLoader = DL.DataReader(10, 13, 13, 10, experiment_no=0)
    traj_new, act_new, goal_new, _ = dataLoader.ReadOneGame("ToMnetF/data/Saved Games/demo/test16.txt")
    #traj_new, act_new, goal_new, _ = dataLoader.ReadOneGame("ToMnetF/data/Saved Games/demo/test29.txt")
    #traj_new, act_new, goal_new, _ = dataLoader.ReadOneGame("ToMnetF/data/Saved Games/demo/test46.txt")
    new_MDP_curr = traj_new[:6, :, :, 0]
    new_MDP_act = act_new

    # get old
    game_past, act_past, goal_past, _ = dataLoader.ReadOneGame("ToMnetF/data/Saved Games/Experiment1/test13.txt")
    trajectories_past, current_states_past, actions_past, goals_past = dataLoader.generateDataFromGame(game_past, act_past, goal_past)
    past_MDP_traj = trajectories_past[-1][:, :, :, :10] #games cant be longer than 10 steps

    width = 13
    height = 13

    def predict(model, past_MDP_trajectories, new_MDP_curr, new_actual_dir):
        
        toMnet = model
        traj = torch.tensor(past_MDP_trajectories, dtype=torch.float32).unsqueeze(dim=0)
        curr = torch.tensor(new_MDP_curr, dtype=torch.float32).unsqueeze(dim=0)
        
        
        simple_map = np.zeros((13, 13))
        predictions = []
        player_position_prime = (0, 0)
        for i in range(len(new_actual_dir)):
            with torch.no_grad():
                prediction = toMnet([traj, curr])
                predictions.append(torch.argmax(prediction, dim=1).item())
                
                # Now we need to move agent in time to pass next curr based on predicted step
                
                
                #  1walls + 1player + 4goals + 4actions
                
                # Put walls
                walls_layer = curr.squeeze(dim=0)[0,...]
                #(6, 13, 13)
                
                if i == 0:
                    for row in range(width):
                        for col in range(height):
                            if walls_layer[row, col] == 0:
                                simple_map[row, col] = 1
            

                player_layer = curr.squeeze(dim=0)[1,...]
                for row in range(width):
                    for col in range(height):
                        if player_layer[row, col] == 1:
                            old_position = [row, col]            
                            player_position = [row, col]

                            if i == 0:
                                simple_map[row, col] = 10
                
                
                if predictions[-1] == 0:    player_position[0] = player_position[0] - 1
                elif predictions[-1] == 1:  player_position[1] = player_position[1] + 1
                elif predictions[-1] == 2:  player_position[0] = player_position[0] + 1
                elif predictions[-1] == 3:  player_position[1] = player_position[1] - 1
                
                if player_position[0] > 13-1: player_position[0] = 13-1
                if player_position[0] < 0: player_position[0] = 0
                if player_position[1] > 13-1: player_position[1] = 13-1
                if player_position[1] < 0: player_position[1] = 0

                new_player_map = torch.zeros(size=(13, 13))
                new_player_map[player_position[0], player_position[1]] = 1
                
                

                action_map = torch.zeros(size=(4, 13, 13))
                action_map[predictions[-1], old_position[0], old_position[1]] = 1
                
                
                # Put goals
                goal_layer = curr.squeeze(dim=0)[2:6,...]
            
                
                if i == 0:
                    for row in range(width):
                        for col in range(height):
                            if goal_layer[0 ,row, col] == 1:
                                simple_map[row, col] = 2
                            elif goal_layer[1, row, col] == 1:
                                simple_map[row, col] = 3
                            elif goal_layer[2, row, col] == 1:
                                simple_map[row, col] = 4
                            elif goal_layer[3, row, col] == 1:
                                simple_map[row, col] = 5
                
                
                curr = torch.concat((walls_layer.unsqueeze(dim=0), new_player_map.unsqueeze(dim=0), goal_layer), dim=0)
                curr = curr.unsqueeze(dim=0).clone().detach()
        
          
        return new_actual_dir, predictions[:len(new_actual_dir)], simple_map   
                
                
    actual, predictions, simple_map = predict(model, past_MDP_traj, new_MDP_curr, new_MDP_act)            


    def RenderActualPredictionMap(actual, predictions, simple_map):
        player_pos = np.where(simple_map==10)
        y = player_pos[0][0] #map is transposed
        x = player_pos[1][0] #map is transposed
        player_pos = (x, y)
        player_pos_pred = (x, y)
        actual_trajectory = []
        predicted_trajectory = []

        if actual is None:
            actual = []
            predictions = []
            gsd.drawMap(13, 13, simple_map=simple_map.T, player_position=None,
                                                init=True, actual=actual_trajectory,
                                                predictions=predicted_trajectory)
        if actual is not None:
            for action in actual:
                player_pos = gsd.newCords([player_pos[0], player_pos[1]], action)
                actual_trajectory.append(player_pos)

            for action in predictions:
                player_pos_pred = gsd.newCords([player_pos_pred[0], player_pos_pred[1]], action)
                predicted_trajectory.append(player_pos_pred)

        for idx, action in enumerate(actual):

            if idx == 0: #init
                posX, posY = gsd.drawMap(13, 13, simple_map=simple_map.T, player_position=None,
                                                init=True, actual=actual_trajectory,
                                                predictions=predicted_trajectory)
                player_pos = gsd.newCords([posX, posY], action)

            else:
                posX, posY = gsd.drawMap(13, 13, simple_map=simple_map.T, player_position=player_pos,
                                                init=False, actual=actual_trajectory)
                player_pos = gsd.newCords(player_pos, action)
                

    if len(set(predictions)) != 1 or set(predictions).pop() == 1:
        """
            PS: How to read Actions:
            0 - UP
            1 - RIGHT
            2 - DOWN
            3 - LEFT
        """
        print("Steps: \n 0 - Up \n 1 - Right \n 2 - Down \n 3 - Left")
        print(f'Actual: {actual}')
        print(f"Predictions: {predictions[:len(predictions)]}")
        RenderActualPredictionMap(actual, predictions, simple_map)
        