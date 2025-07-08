import numpy as np
import heapq
import re
import os
import sys

sys.path.append("../experiment1/")

"""
A* Agent for ToMnetF experiments
Original code from https://github.com/Nik-Kras/ToMnet-N
@Author Nikita Krasnytskyi
@Modified by Filip Borowiak
"""

adjacent_squares = (
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
)


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f


class AgentStar:
    """
    A* Agent that navigates the maze optimally to reach goals
    """

    def __init__(self, env, sight, observability="partial", consume_goals=1):
        self.env = env
        self.consume_goals = consume_goals
        self.observability = observability
        if self.observability == "partial":
            self.sight = min(sight, self.env.height)
        else:
            self.sight = None
        self.position = env.players_position
        self.memory = np.full((env.width, env.height), None)
        self.trajectory = []
        self.position_trajectory = []
        self.step_picked_goal = []
        self.picked_list = []
        self.goal_found = None
        self.max_goal = self.get_highest_goal(self.picked_list)
        self.picked_goal = False

    """
    PS: How to read Actions:
    0 - UP
    1 - RIGHT
    2 - DOWN
    3 - LEFT
    """

    def get_highest_goal(self, ignore_list):
        max_value = 0
        result_point = None
        for key, value in self.env.GoalValue.items():
            if max_value < value and key not in ignore_list:
                max_value = value
                result_point = key

        return result_point

    def get_highest_goal_from_memory(self, ignore_list):
        value = 0
        result_point = None
        for row in self.memory:
            for point in row:
                if (
                    point in self.env.GoalValue
                    and value < self.env.GoalValue[point]
                    and point not in ignore_list
                ):
                    value = self.env.GoalValue[point]
                    result_point = point

        return result_point

    def chose_action(self, observability="partial"):
        if observability == "full" or self.picked_goal:
            self.goal_found = self.get_highest_goal_from_memory([])

        result = self.astar(observability=observability)

        ignore_list = []

        if self.goal_found is None:
            while result == -1:
                self.goal_found = self.get_highest_goal_from_memory(ignore_list)
                """ we need this to solve this problem
                # - #
                # O #
                D A #
                """
                if self.goal_found is None:
                    self.goal_found = self.env.objectsEnum["Wall"]

                ignore_list.append(self.goal_found)
                result = self.astar(observability=observability)

        # REMOVED: Duplicate call - already handled in on_pickup()
        # if self.env.goal_picked != 0:
        #     self.step_picked_goal.append(len(self.trajectory) - 1)
        self.position_trajectory.append(self.position)
        self.trajectory.append(result)
        return result

    def on_pickup(self, reward):
        self.memory[self.env.players_position[0], self.env.players_position[1]] = (
            self.env.objectsEnum["Path"]
        )
        self.picked_goal = True

    # Astar-like algorithm
    def astar(self, observability="partial"):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""

        # Create start and end node
        start_node = Node(None, position=tuple(self.position))
        start_node.g = start_node.h = start_node.f = 0

        if observability == "full" and not (self.goal_found is None):
            goal_row = goal_col = 0
            for i in range(self.env.height):
                for j in range(self.env.width):
                    if self.memory[i, j] == self.goal_found:
                        goal_row = i
                        goal_col = j
            goal_position = [goal_row, goal_col]
            end_node = Node(None, position=tuple(goal_position))
        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Heapify the open_list and Add the start node
        heapq.heapify(open_list)
        heapq.heappush(open_list, start_node)

        # Loop until you find the end
        while len(open_list) > 0:
            # Get the current node
            current_node = heapq.heappop(open_list)
            if not (current_node in closed_list):
                closed_list.append(current_node)

            point_obj = self.memory[current_node.position[0], current_node.position[1]]
            # Found the unexplored cell or it's goal
            if point_obj == self.goal_found or point_obj == self.max_goal:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent

                try:
                    first_node = path[-2]
                except:
                    return
                if first_node[1] < self.position[1]:
                    return 3
                elif first_node[1] > self.position[1]:
                    return 1
                elif first_node[0] < self.position[0]:
                    return 0
                elif first_node[0] > self.position[0]:
                    return 2

                return -1

            # Generate children
            children = []
            for new_position in adjacent_squares:  # Adjacent squares

                # Get node position
                node_position = (
                    current_node.position[0] + new_position[0],
                    current_node.position[1] + new_position[1],
                )

                # Make sure within range
                if (
                    node_position[0] > (self.env.width - 1)
                    or node_position[0] < 0
                    or node_position[1] > (self.env.height - 1)
                    or node_position[1] < 0
                ):
                    continue

                # Make sure we don't wall into walls
                point_obj = self.memory[node_position[0], node_position[1]]
                if not (
                    point_obj == self.env.objectsEnum["Path"]
                    or point_obj == self.goal_found
                    or point_obj == self.max_goal
                ):
                    continue

                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                # Child is on the closed list
                if (
                    len(
                        [
                            closed_child
                            for closed_child in closed_list
                            if closed_child == child
                        ]
                    )
                    > 0
                ):
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + 1

                # Use ASTAR for path optimization
                # For Full-observability
                if observability == "full":
                    child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                        (child.position[1] - end_node.position[1]) ** 2
                    )
                else:
                    child.h = 0
                child.f = child.g + child.h

                # Child is already in the open list
                if (
                    len(
                        [
                            open_node
                            for open_node in open_list
                            if child.position == open_node.position
                            and child.g > open_node.g
                        ]
                    )
                    > 0
                ):
                    continue

                # Add the child to the open list
                heapq.heappush(open_list, child)

        return -1

    def update_world_observation(self):
        position, sight_array = self.env.get_sight(self.sight, self.observability)
        self.position = position

        if self.observability == "full":
            self.memory = sight_array
        else:

            half_sight = int(self.sight * 0.5)
            for i in range(self.sight):
                for j in range(self.sight):
                    x = i + position[0] - half_sight
                    y = j + position[1] - half_sight

                    sight_elem = sight_array[i, j]
                    if (
                        -1 < x < self.env.width
                        and -1 < y < self.env.height
                        and sight_elem is not None
                    ):
                        self.memory[x, y] = sight_elem

    def render(self):
        # 2. Draw a Map
        graph = ""
        for row in range(self.env.width):
            row_string = ""
            for col in range(self.env.height):

                # Draw player
                if self.position == [row, col]:
                    row_string += " \u25cb "  # u" \u25CC "

                # Draw walls, paths and goals
                else:
                    if self.memory[row, col] is None:
                        row_string += " ? "  # Unexplored
                    elif self.memory[row, col] == self.env.objectsEnum["Wall"]:
                        row_string += " # "  # Wall
                    elif self.memory[row, col] == self.env.objectsEnum["Path"]:
                        row_string += " - "  # Path
                    elif self.memory[row, col] == self.env.objectsEnum["Goal A"]:
                        row_string += " A "  # Goal 1
                    elif self.memory[row, col] == self.env.objectsEnum["Goal B"]:
                        row_string += " B "  # Goal 2
                    elif self.memory[row, col] == self.env.objectsEnum["Goal C"]:
                        row_string += " C "  # Goal 3
                    elif self.memory[row, col] == self.env.objectsEnum["Goal D"]:
                        row_string += " D "  # Goal 4
                    else:
                        print("ERROR: Incorrect map value! Position: ", row, ", ", col)

            row_string += "\n"
            graph += row_string
        print(graph)
        return graph

    """
        A function saves a game: Maze (walls, goals, player), 
        Consumed Goal, Length of trajectory and each trajectory step to .txt file
    """

    def save_game(self, name="experiment1", base_dir="../../data"):

        # REMOVED: Duplicate call - already handled in on_pickup()
        # if self.env.goal_picked != 0:
        #     self.step_picked_goal.append(len(self.trajectory) - 1)

        # Get the path to folder
        gf = os.path.join(base_dir, name)  # path to games folder
        os.makedirs(gf, exist_ok=True)

        files = os.listdir(gf)
        r = re.compile(".*.txt")
        files = list(filter(r.match, files))

        # Chose the number for the new name
        # print("There are files in the folder: ", files)
        max_number = 0
        for file in files:
            if max_number < int(file[4:-4]):
                max_number = int(file[4:-4])

        # print("The max current number is: ", max_number)
        new_name_number = max_number + 1

        # Save the Game line by line
        new_file_path = os.path.join(gf, "test" + str(new_name_number) + ".txt")
        # realmap = self.render()

        with open(new_file_path, "w") as f:
            f.write("Maze:\n")

            # Save the Maze (Walls, Goals, Player)
            wall_line = "#" * (self.env.height + 2)
            f.write(wall_line + "\n")
            for i in range(self.env.width):
                f.write(self.env.init_map[i] + "\n")
            f.write(wall_line + "\n")
            for i, goal in enumerate(self.env.consumed_goal):
                f.write("Goal Consumed #" + str(i + 1) + " : " + goal + "\n")
            f.write("Trajectory length: " + str(len(self.trajectory)) + "\n")

            # Save moves
            for i in range(len(self.trajectory)):
                msg = (
                    str(self.position_trajectory[i])
                    + " : "
                    + str(self.trajectory[i])
                    + " : "
                )
                picked_bool = False
                for j in range(len(self.step_picked_goal)):
                    if self.step_picked_goal[j] == i:
                        msg = (
                            msg + self.env.consumed_goal[j]
                        )  # If consumed First goal here - mention which
                        picked_bool = True
                if not picked_bool:
                    msg = msg + "X"  # If didn't consume - put X
                f.write(msg + "\n")

            f.close()


class ValueAgent:
    """
    Value-based agent with stochastic action selection using value iteration
    Based on GoalDirectedAgent but adapted for ToMnetF environment
    """

    def __init__(self, env, sight, observability="partial", consume_goals=1,
                 movement_cost=0.01, wall_penalty=0.05, gamma=0.99, temperature=0.1):
        self.env = env
        self.consume_goals = consume_goals
        self.observability = observability
        if self.observability == "partial":
            self.sight = min(sight, self.env.height)
        else:
            self.sight = None
        self.position = env.players_position
        self.memory = np.full((env.width, env.height), None)
        self.trajectory = []
        self.position_trajectory = []
        self.step_picked_goal = []
        self.picked_list = []
        self.goal_found = None
        self.picked_goal = False
        
        # Value iteration parameters
        self.movement_cost = movement_cost
        self.wall_penalty = wall_penalty
        self.gamma = gamma
        self.temperature = temperature
        
        # Get reward preferences from environment
        self.rewards = np.array(env.goal_rewards, dtype=np.float32)
        
        # Value function and policy
        self.value_function = None
        self.policy = None
        self.converged = False
        
        # Action mapping: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def get_highest_goal(self, ignore_list):
        max_value = 0
        result_point = None
        for key, value in self.env.GoalValue.items():
            if max_value < value and key not in ignore_list:
                max_value = value
                result_point = key
        return result_point

    def get_highest_goal_from_memory(self, ignore_list):
        value = 0
        result_point = None
        for row in self.memory:
            for point in row:
                if (
                    point in self.env.GoalValue
                    and value < self.env.GoalValue[point]
                    and point not in ignore_list
                ):
                    value = self.env.GoalValue[point]
                    result_point = point
        return result_point

    def plan_value_iteration(self, max_iterations=100, convergence_threshold=0.01):
        """
        Run value iteration to compute optimal policy for current environment state
        """
        width, height = self.env.width, self.env.height
        n_actions = 4
        
        # Initialize value function and policy
        self.value_function = np.zeros((width, height))
        self.policy = np.ones((width, height, n_actions)) / n_actions
        
        # Create valid state mask from memory
        valid_states = np.zeros((width, height), dtype=bool)
        for i in range(width):
            for j in range(height):
                if self.memory[i, j] is not None and self.memory[i, j] != self.env.objectsEnum["Wall"]:
                    valid_states[i, j] = True
        
        for iteration in range(max_iterations):
            old_values = self.value_function.copy()
            
            # Value iteration update
            for i in range(width):
                for j in range(height):
                    if not valid_states[i, j]:
                        continue
                    
                    # Compute Q-values for each action
                    q_values = np.zeros(n_actions)
                    for action in range(n_actions):
                        q_values[action] = self._evaluate_action_from_memory((i, j), action)
                    
                    # Update value function
                    self.value_function[i, j] = np.max(q_values)
                    
                    # Update policy with softmax
                    if self.temperature > 0:
                        q_values_clipped = np.clip(q_values, -100, 100)
                        exp_q = np.exp(q_values_clipped / self.temperature)
                        self.policy[i, j] = exp_q / np.sum(exp_q)
                    else:
                        # Deterministic policy
                        self.policy[i, j] = 0
                        best_action = np.argmax(q_values)
                        self.policy[i, j, best_action] = 1.0
            
            # Check convergence
            if np.max(np.abs(self.value_function - old_values)) < convergence_threshold:
                self.converged = True
                break
        
        if not self.converged:
            print(f"Warning: Value iteration did not converge after {max_iterations} iterations")

    def _evaluate_action_from_memory(self, pos, action):
        """Evaluate expected value of taking action from position using memory"""
        i, j = pos
        delta = self.actions[action]
        new_pos = (i + delta[0], j + delta[1])
        
        # Base movement cost
        reward = -self.movement_cost
        
        # Check bounds
        if (new_pos[0] < 0 or new_pos[0] >= self.env.width or 
            new_pos[1] < 0 or new_pos[1] >= self.env.height):
            reward -= self.wall_penalty
            next_value = self.gamma * self.value_function[i, j]
        else:
            memory_value = self.memory[new_pos[0], new_pos[1]]
            
            # Check if it's a wall
            if memory_value == self.env.objectsEnum["Wall"]:
                reward -= self.wall_penalty
                next_value = self.gamma * self.value_function[i, j]
            else:
                # Check if it's a goal
                if memory_value in self.env.GoalValue:
                    goal_reward = self.env.GoalValue[memory_value]
                    reward += goal_reward
                    next_value = 0  # Terminal state
                else:
                    next_value = self.gamma * self.value_function[new_pos[0], new_pos[1]]
        
        return reward + next_value

    def chose_action(self, observability="partial"):
        """Choose action using stochastic policy from value iteration"""
        
        # Update memory and plan if needed
        self.update_world_observation()
        
        # Plan policy based on current memory
        self.plan_value_iteration()
        
        # Get current position
        current_pos = self.position
        
        # Get action probabilities from policy
        if (self.policy is not None and 
            0 <= current_pos[0] < self.env.width and 
            0 <= current_pos[1] < self.env.height):
            
            action_probs = self.policy[current_pos[0], current_pos[1]].copy()
            
            # Ensure valid probability distribution
            if np.any(np.isnan(action_probs)) or np.sum(action_probs) == 0:
                action_probs = np.ones(4) / 4
            else:
                action_probs = action_probs / np.sum(action_probs)
            
            # Sample action stochastically
            action = np.random.choice(4, p=action_probs)
        else:
            # Fallback to random action
            action = np.random.choice(4)
        
        # Record trajectory
        self.position_trajectory.append(self.position.copy())
        self.trajectory.append(action)
        
        return action

    def update_world_observation(self):
        position, sight_array = self.env.get_sight(self.sight, self.observability)
        self.position = position

        if self.observability == "full":
            self.memory = sight_array
        else:
            half_sight = int(self.sight * 0.5)
            for i in range(self.sight):
                for j in range(self.sight):
                    x = i + position[0] - half_sight
                    y = j + position[1] - half_sight

                    sight_elem = sight_array[i, j]
                    if (
                        -1 < x < self.env.width
                        and -1 < y < self.env.height
                        and sight_elem is not None
                    ):
                        self.memory[x, y] = sight_elem

    def on_pickup(self, reward):
        self.memory[self.env.players_position[0], self.env.players_position[1]] = (
            self.env.objectsEnum["Path"]
        )
        self.picked_goal = True
        self.step_picked_goal.append(len(self.trajectory) - 1)

    def save_game(self, name="experiment5", base_dir="../../data"):
        # Use the same save format as AgentStar
        import re
        import os
        
        # REMOVED: Duplicate call - already handled in on_pickup()
        # if self.env.goal_picked != 0:
        #     self.step_picked_goal.append(len(self.trajectory) - 1)

        # Get the path to folder
        gf = os.path.join(base_dir, name)
        os.makedirs(gf, exist_ok=True)

        files = os.listdir(gf)
        r = re.compile(".*.txt")
        files = list(filter(r.match, files))

        # Choose the number for the new name
        max_number = 0
        for file in files:
            if max_number < int(file[4:-4]):
                max_number = int(file[4:-4])

        new_name_number = max_number + 1

        # Save the Game line by line
        new_file_path = os.path.join(gf, "test" + str(new_name_number) + ".txt")

        with open(new_file_path, "w") as f:
            f.write("Maze:\n")

            # Save the Maze (Walls, Goals, Player)
            wall_line = "#" * (self.env.height + 2)
            f.write(wall_line + "\n")
            for i in range(self.env.width):
                f.write(self.env.init_map[i] + "\n")
            f.write(wall_line + "\n")
            for i, goal in enumerate(self.env.consumed_goal):
                f.write("Goal Consumed #" + str(i + 1) + " : " + goal + "\n")
            f.write("Trajectory length: " + str(len(self.trajectory)) + "\n")

            # Save moves
            for i in range(len(self.trajectory)):
                msg = (
                    str(self.position_trajectory[i])
                    + " : "
                    + str(self.trajectory[i])
                    + " : "
                )
                picked_bool = False
                for j in range(len(self.step_picked_goal)):
                    if self.step_picked_goal[j] == i:
                        msg = (
                            msg + self.env.consumed_goal[j]
                        )  # If consumed First goal here - mention which
                        picked_bool = True
                if not picked_bool:
                    msg = msg + "X"  # If didn't consume - put X
                f.write(msg + "\n")

            f.close()


class RandomAgent:
    """
    Random agent for comparison
    """

    def __init__(self, env, sight, observability="partial"):
        self.env = env
        self.observability = observability
        self.sight = sight
        self.position = env.players_position
        self.trajectory = []
        self.position_trajectory = []
        self.step_picked_goal = []

    def chose_action(self, observability="partial"):
        action = np.random.choice(4)
        self.position_trajectory.append(self.position)
        self.trajectory.append(action)
        return action

    def update_world_observation(self):
        position, sight_array = self.env.get_sight(self.sight, self.observability)
        self.position = position

    def on_pickup(self, reward):
        self.step_picked_goal.append(len(self.trajectory) - 1)

    def save_game(self, name="experiment1", base_dir="../../data"):
        # Similar to AgentStar but simplified
        pass
