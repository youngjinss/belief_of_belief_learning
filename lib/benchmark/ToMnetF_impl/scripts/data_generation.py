import os
import numpy as np
import re
import matplotlib.pyplot as plt
import torch
import pickle

"""
Agnostic data processing and loading utilities for ToMnet
Combined from FDataLoader.py and FDataProcessor.py
@author: Chuang, Yun-Shiuan; Edwinn
@modified by: Filip Borowiak
"""


class DataReader:
    def __init__(self, time_step, w, h, d, experiment_no):

        self.EXPERIMENT_NO = experiment_no
        self.MAX_TRAJECTORY_SIZE = time_step
        self.MAZE_WIDTH = w  # 13
        self.MAZE_HEIGHT = h  # 13
        self.MAZE_DEPTH_TRAJECTORY = (
            d  # 10 (1 - player, 1 - walls, 4 - objects, 4 - directions)
        )

    def check_ifCollected(self, x):

        if x[-2] == "A" or x[-2] == "B" or x[-2] == "C" or x[-2] == "D":
            return True

    def ReadOneGame(self, filename):
        f = open(filename, "r")
        # lines 1: 15 - map
        map = []
        consumed = []
        trajectory_length = []
        trajectory = {}
        actions = []
        steps = []
        consumption_labels = None
        sr_maps = {}

        traj = np.empty(
            (1, self.MAZE_HEIGHT, self.MAZE_WIDTH, self.MAZE_DEPTH_TRAJECTORY)
        )

        act = np.empty(1, dtype=np.int8)
        goal = np.empty(1, dtype=np.int8)

        for idx, x in enumerate(f):
            temp_pos = []
            temp_row = []

            if idx >= 2 and idx <= 14:
                for idx_2, c in enumerate(x):
                    if idx_2 >= 1 and idx_2 <= 13:
                        if c == "#":
                            temp_row.append(0)
                        elif c == "A":
                            temp_row.append(2)
                        elif c == "B":
                            temp_row.append(3)
                        elif c == "C":
                            temp_row.append(4)
                        elif c == "D":
                            temp_row.append(5)
                        elif c == "-":
                            temp_row.append(1)
                        elif c == "O":
                            temp_row.append(10)

                map.append(temp_row)

            if idx == 16:
                consumed.append(x)
            if idx == 17:
                trajectory_length.append(x)

            # Parse consumption labels if present
            if x.startswith("Consumption Labels:"):
                consumption_str = x.split(":")[1].strip()
                consumption_labels = np.array(
                    [float(val) for val in consumption_str.split(",")], dtype=np.float32
                )

            # Parse SR maps if present (new sparse format)
            if x.startswith("SR_gamma_"):
                gamma_str = x.split(":")[0].split("_")[-1]
                sr_str = x.split(":")[1].strip()

                # Initialize empty SR map
                sr_map = np.zeros((self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.float32)

                # Parse sparse format: "x,y:value;x2,y2:value2;..."
                if sr_str.strip():  # Only if not empty
                    pairs = sr_str.split(";")
                    for pair in pairs:
                        if ":" in pair:
                            pos_str, val_str = pair.split(":")
                            if "," in pos_str:
                                x, y = map(int, pos_str.split(","))
                                value = float(val_str)
                                if (
                                    0 <= x < self.MAZE_WIDTH
                                    and 0 <= y < self.MAZE_HEIGHT
                                ):
                                    sr_map[x, y] = value

                sr_maps[gamma_str] = sr_map

            if (
                idx >= 18
                and not x.startswith("Consumption Labels:")
                and not x.startswith("SR_gamma_")
                and not x.startswith("SR_Data_Per_Timestep:")
                and not x.startswith("Timestep_")
            ):

                # Parse trajectory lines: [x, y] : action : goal
                import re

                pattern = r"\[(\d+),\s*(\d+)\]\s*:\s*(\d+)\s*:\s*(\w+)"
                match = re.match(pattern, x)

                if match:
                    pos_x = int(match.group(1))
                    pos_y = int(match.group(2))
                    temp_traj = int(match.group(3))
                    temp_pos = (pos_x, pos_y)

                    actions.append(temp_traj)
                else:
                    # Skip malformed lines
                    continue

                trajectory[temp_pos] = temp_traj

        map = np.array(map)  # Map
        consumed = consumed[0][19]  # Consumed goal

        # Plane for obstacles - static
        np_obstacles = np.where(map == 0, 1, 0).astype(np.int8)  # if wall then 1 else 0

        # Plane for agent's initial position
        np_agent = np.where(map == 10, 1, 0).astype(np.int8)

        #        A  B  C  D
        goals = [2, 3, 4, 5]
        np_targets = np.zeros((4, self.MAZE_WIDTH, self.MAZE_HEIGHT))
        for target, i in zip(goals, range(len(goals))):
            np_targets[
                i,
                :,
                :,
            ] = np.where(
                map == target, 1, 0
            ).astype(np.int8)

        directions = {"Action:0": 0, "Action:1": 0, "Action:2": 0, "Action:3": 0}

        for idx, (key, val) in enumerate(trajectory.items()):
            posX = key[0]
            posY = key[1]
            temp_trajectory = val
            if idx == 0:  # first values are init numpy values - replace
                goal[0] = self.goal_sym_to_num(consumed)
                act[0] = val
            else:

                act = np.append(act, val)
                goal = np.append(goal, self.goal_sym_to_num(consumed))

                np_agent = np.zeros(
                    shape=(self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.int8
                )
                np_agent[posX, posY] = 1

            # Make tensor traj
            # np_actions = np.zeros((self.MAZE_WIDTH, self.MAZE_HEIGHT, 5), dtype=np.int8)
            np_actions = np.zeros((4, self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.int8)
            np_actions[val, int(posX), int(posY)] = 1  # update trajectory taken
            if val == 0:
                directions["Action:0"] += 1
            elif val == 1:
                directions["Action:1"] += 1
            elif val == 2:
                directions["Action:2"] += 1
            elif val == 3:
                directions["Action:3"] += 1

            np_obstacles1 = np.expand_dims(np_obstacles, 0)
            np_agent1 = np.expand_dims(np_agent, 0)

            tensor = np.concatenate(
                (np_obstacles1, np_agent1, np_targets, np_actions)
            )  # (1walls + 1player + 4goals + 4actions)

            # Reshape from (depth, height, width) to (height, width, depth)
            tensor = np.transpose(tensor, (1, 2, 0))  # (height, width, depth)

            steps.append(tensor)  # each step (record) is one decision data

        # Stack steps along time dimension
        if steps:
            traj = np.stack(steps, axis=0)  # (time_steps, height, width, depth)

        # traj = torch.tensor(steps)
        # If SR maps were loaded, convert to numpy array with 3 channels
        if sr_maps:
            sr_array = np.zeros(
                (3, self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.float32
            )
            for i, gamma in enumerate(["0.5", "0.9", "0.99"]):
                if gamma in sr_maps:
                    sr_array[i] = sr_maps[gamma]
        else:
            sr_array = None

        return traj, act, goal, directions, consumption_labels, sr_array

        # print(f"map size: {map.shape}\n")
        # print(map)

        # trajectory_length = trajectory_length[0][19:]

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
        print("Games names: ", len(files))

        # Save all trajectories and labels
        trajectories = (
            []
        )  # np.empty([1, self.MAZE_WIDTH, self.MAZE_HEIGHT, self.MAZE_DEPTH_TRAJECTORY])
        actions = []  # np.empty(1)
        labels = []  # np.empty(1)
        consumption_labels_list = []
        sr_maps_list = []
        """
            How to read Actions:
            0 - UP
            1 - RIGHT
            2 - DOWN
            3 - LEFT
        """
        directions_total = {"Action:0": 0, "Action:1": 0, "Action:2": 0, "Action:3": 0}
        directions_mapping = {
            "Action Up": 0,
            "Action Right": 1,
            "Action Down": 2,
            "Action Left": 3,
        }

        # ------------------------------------------------------------------
        # 1. Load each game one by one
        # ------------------------------------------------------------------
        j = 0  # for tracking progress (%)
        for i, file in enumerate(files):

            # Read one game
            traj, act, goal, directions, consumption_labels, sr_map = self.ReadOneGame(
                filename=os.path.join(directory, file)
            )

            directions_total["Action:0"] += directions["Action:0"]
            directions_total["Action:1"] += directions["Action:1"]
            directions_total["Action:2"] += directions["Action:2"]
            directions_total["Action:3"] += directions["Action:3"]

            # Append a game to data
            trajectories.append(traj)
            actions.append(act)
            labels.append(goal)
            consumption_labels_list.append(consumption_labels)
            sr_maps_list.append(sr_map)

            # Keep track on progress
            if i >= int(np.ceil(j * Nfraction / 100)) - 1:
                print("Parsed " + str(j) + "%")
                j += 10
        print("----")

        print("Augment data. One game creates many training samples!")

        data_trajectories = []
        data_current_state = []
        data_actions = []
        data_labels = []
        data_consumption_labels = []
        data_sr_maps = []
        j = 0  # for tracking progress (%)

        # Process Game-per-Game
        for i in range(Nfraction):

            # Consider only games with more than 6 moves
            if trajectories[i].shape[0] < 6:
                continue

            # Prepare data from one game
            # The dimensions differ, so only list is applicable (no numpy arrays)
            (
                data_trajectories1,
                data_current_state1,
                data_actions1,
                data_labels1,
                consumption_labels1,
                sr_maps1,
            ) = self.generateDataFromGame(
                trajectories=trajectories[i],
                actions=actions[i],
                labels=labels[i],
                consumption_labels=consumption_labels_list[i],
                sr_map=sr_maps_list[i],
            )

            # Append to a single structure
            data_trajectories.append(data_trajectories1)
            data_current_state.append(data_current_state1)
            data_actions.append(data_actions1)
            data_labels.append(data_labels1)
            data_consumption_labels.append(consumption_labels1)
            data_sr_maps.append(sr_maps1)

            # Keep track on progress
            if i >= int(np.ceil(j * Nfraction / 100)) - 1:
                print("Augmented data " + str(j) + "%")
                j += 10

        print("----")

        # data_trajectories1 shape is ()
        all_games = {
            "traj_history": data_trajectories,
            "current_state_history": data_current_state,
            "actions_history": data_actions,
            "labels_history": data_labels,
            "consumption_labels_history": data_consumption_labels,
            "sr_maps_history": data_sr_maps,
        }

        print(f"Directions count: {directions_total}")
        print(f"Directions mapping: {directions_mapping}")
        names = [key for key, _ in directions_mapping.items()]
        values = [val for _, val in directions_total.items()]

        # Create result directory for plots
        plt.bar(names, values)
        plt.grid(True)
        plt.legend()
        plt.title("Distribution of Actions")
        plt.savefig(f"{directory}/action_distribution")
        plt.show()

        return all_games

    def generateDataFromGame(
        self, trajectories, actions, labels, consumption_labels, sr_map
    ):

        # Make full data from a game
        data_trajectories = []
        data_current_state = []
        data_actions = []
        data_labels = []
        data_consumption_labels = []
        data_sr_maps = []

        MIN_ACTIONS = 6
        for i in range(MIN_ACTIONS, trajectories.shape[0]):
            data_trajectories.append(
                trajectories[0:i, :, :, :]
            )  # Trajectory to the state (time_steps, height, width, depth)
            data_current_state.append(
                trajectories[i, :, :, 0:6]
            )  # Current state # (height, width, 6channels: 1walls + 1player + 4goals)
            data_actions.append(actions[i, ...])  # Next Action
            data_labels.append(labels[i, ...])  # Consumed Goal

            # For consumption labels and SR maps, use the same values for all timesteps
            # since they represent the final state of the episode
            if consumption_labels is not None:
                data_consumption_labels.append(consumption_labels)
            else:
                # Default to zeros if not available (for backward compatibility)
                data_consumption_labels.append(np.zeros(4, dtype=np.float32))

            if sr_map is not None:
                data_sr_maps.append(sr_map)
            else:
                # Default to zeros if not available (for backward compatibility)
                data_sr_maps.append(
                    np.zeros((3, self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.float32)
                )

        return (
            data_trajectories,
            data_current_state,
            data_actions,
            data_labels,
            data_consumption_labels,
            data_sr_maps,
        )

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


class DataProcessor:

    def __init__(self, time_step, w, h, d):
        self.MAX_TRAJECTORY_SIZE = time_step  # 20-50
        self.MAZE_WIDTH = w  # 13
        self.MAZE_HEIGHT = h  # 13
        self.MAZE_DEPTH = d  # 11 (1player + 1wall + 4goals + 5 actions = 11)

    def zeroPadding(self, max_elements, all_games):

        # all_games = {
        # "traj_history": data_trajectories,           # Trajectories until state
        # "traj_history_zp": traj_history_zp           # Trajectory with Zero Padding
        # "current_state_history": data_current_state, # (1walls + 1player + 4goals)
        # "actions_history": data_actions,             # actions
        # "labels_history": data_labels                # goals
        # }
        uniform_shape = (
            1,
            max_elements,
            self.MAZE_HEIGHT,
            self.MAZE_WIDTH,
            self.MAZE_DEPTH,
        )

        zero_padded_trajectories = []  # ndarray, not list
        unfolded_current_states = []
        unfolded_action_history = []
        unfolded_goal_history = []
        unfolded_consumption_labels = []
        unfolded_sr_maps = []
        all_trajectories = all_games["traj_history"]
        all_current_states = all_games["current_state_history"]
        all_actions = all_games["actions_history"]
        all_labels = all_games["labels_history"]
        all_consumption_labels = all_games.get("consumption_labels_history", [])
        all_sr_maps = all_games.get("sr_maps_history", [])
        N_all_games = len(all_trajectories)

        # Go one by one game
        # Where each game consist of many trajectories
        tracker_var = 0
        for i in range(N_all_games):

            traj = all_trajectories[i]
            cur = all_current_states[i]
            act = all_actions[i]
            goal = all_labels[i]
            consumption = (
                all_consumption_labels[i] if i < len(all_consumption_labels) else []
            )
            sr_map = all_sr_maps[i] if i < len(all_sr_maps) else []
            N_traj = len(
                traj
            )  # traj.shape[0]      # Number of trajectories in current game

            for j in range(N_traj):

                ### Init single piece of data from a game -> j = one game
                current_trajectory = traj[j]
                current_state = cur[j]
                current_action = act[j]
                current_goal = goal[j]
                current_consumption = (
                    consumption[j]
                    if j < len(consumption)
                    else np.zeros(4, dtype=np.float32)
                )
                current_sr_map = (
                    sr_map[j]
                    if j < len(sr_map)
                    else np.zeros(
                        (3, self.MAZE_WIDTH, self.MAZE_HEIGHT), dtype=np.float32
                    )
                )

                ### Trajectory
                zero_pad_trajectory = np.zeros(shape=uniform_shape)
                Nt = current_trajectory.shape[
                    0
                ]  # Number of real steps in the trajectory

                # Save game in a bigger array so the rest is filled with zeros
                if Nt > max_elements:
                    zero_pad_trajectory[0, ...] = current_trajectory[
                        -max_elements:, :, :, :
                    ]
                else:
                    zero_pad_trajectory[0, 0:Nt, :, :, :] = current_trajectory

                zero_padded_trajectories.append(zero_pad_trajectory[0, ...])

                ### Current state
                unfolded_current_states.append(current_state)

                ### Action
                unfolded_action_history.append(current_action)

                ### Goal
                unfolded_goal_history.append(current_goal)

                ### Consumption labels
                unfolded_consumption_labels.append(current_consumption)

                ### SR maps
                unfolded_sr_maps.append(current_sr_map)

            # Keep track on progress
            if i >= int(N_all_games * tracker_var / 100) - 2:
                print("Zero-Padded data " + str(tracker_var) + "%")
                tracker_var += 5

        zero_padded_trajectories = np.array(zero_padded_trajectories)
        unfolded_current_states = np.array(unfolded_current_states)
        unfolded_action_history = np.array(unfolded_action_history)
        unfolded_goal_history = np.array(unfolded_goal_history)
        unfolded_consumption_labels = np.array(unfolded_consumption_labels)
        unfolded_sr_maps = np.array(unfolded_sr_maps)

        all_games["traj_history"] = all_trajectories
        all_games["traj_history_zp"] = zero_padded_trajectories
        all_games["current_state_history"] = unfolded_current_states
        all_games["actions_history"] = unfolded_action_history
        all_games["labels_history"] = unfolded_goal_history
        all_games["consumption_labels_history"] = unfolded_consumption_labels
        all_games["sr_maps_history"] = unfolded_sr_maps

        print(f"traj_zp history shape: {all_games['traj_history_zp'].shape}")
        print(f"samples: {all_games['traj_history_zp'].shape[0]}")
        print(f"trajectory size: {all_games['traj_history_zp'].shape}")

        print("Zero Padding was applied!")

        return all_games


def generate_input_data(
    data_dir="../data/experiment1",
    output_dir="../data/experiment1",
    use_percentage=0.9,
    time_step=10,
    height=13,
    width=13,
    depth=10,
    experiment_no=1,
):
    """
    Generate processed input data for ToMnet training

    Args:
        data_dir: Directory containing game txt files
        output_dir: Directory to save processed data
        use_percentage: Percentage of games to use
        time_step: Trajectory size/length
        height: Map height
        width: Map width
        depth: Tensor depth (channels)
        experiment_no: Experiment number
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize data reader and processor
    dl = DataReader(time_step, height, width, depth, experiment_no)
    dp = DataProcessor(time_step, height, width, depth)

    # Load all games
    all_games = dl.LoadAllGames(use_percentage=use_percentage, directory=data_dir)

    # Apply zero padding
    all_games = dp.zeroPadding(time_step, all_games)

    # Extract processed data
    data_trajectories = all_games["traj_history"]
    data_trajectories_zp = all_games["traj_history_zp"]
    data_current_state = all_games["current_state_history"]
    data_actions = all_games["actions_history"]
    data_labels = all_games["labels_history"]
    data_consumption_labels = all_games.get("consumption_labels_history", None)
    data_sr_maps = all_games.get("sr_maps_history", None)

    # Convert to tensors
    data_traj = torch.tensor(data_trajectories_zp, dtype=torch.float32)
    data_curr = torch.tensor(data_current_state, dtype=torch.float32)
    data_act = torch.tensor(data_actions, dtype=torch.float32)
    data_labels = torch.tensor(data_labels, dtype=torch.float32)

    # Convert consumption and SR data if available
    if data_consumption_labels is not None:
        data_consumption = torch.tensor(data_consumption_labels, dtype=torch.float32)
    else:
        data_consumption = None

    if data_sr_maps is not None:
        data_sr = torch.tensor(data_sr_maps, dtype=torch.float32)
    else:
        data_sr = None

    # Save processed data
    processed_data = {
        "data_trajectories": data_traj,
        "data_current_state": data_curr,
        "data_actions": data_act,
        "data_labels": data_labels,
        "data_consumption_labels": data_consumption,
        "data_sr_maps": data_sr,
        "metadata": {
            "time_step": time_step,
            "height": height,
            "width": width,
            "depth": depth,
            "experiment_no": experiment_no,
            "use_percentage": use_percentage,
            "n_samples": data_traj.shape[0],
        },
    }

    output_file = os.path.join(output_dir, f"processed_data_exp{experiment_no}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(processed_data, f)

    print(f"Processed data saved to: {output_file}")
    print(f"Number of samples: {data_traj.shape[0]}")
    print(f"Trajectory shape: {data_traj.shape}")
    print(f"Current state shape: {data_curr.shape}")

    return processed_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate processed input data for ToMnet training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/experiment1",
        help="Directory containing game txt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/experiment1",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--use_percentage",
        type=float,
        default=0.9,
        help="Percentage of games to use (0.0-1.0)",
    )
    parser.add_argument(
        "--time_step", type=int, default=10, help="Trajectory size/length"
    )
    parser.add_argument("--height", type=int, default=13, help="Map height")
    parser.add_argument("--width", type=int, default=13, help="Map width")
    parser.add_argument("--depth", type=int, default=10, help="Tensor depth (channels)")
    parser.add_argument(
        "--experiment_no", type=int, default=1, help="Experiment number"
    )

    args = parser.parse_args()

    # Generate processed data
    processed_data = generate_input_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_percentage=args.use_percentage,
        time_step=args.time_step,
        height=args.height,
        width=args.width,
        depth=args.depth,
        experiment_no=args.experiment_no,
    )
