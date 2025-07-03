from Maze import MazeGenerator
import numpy as np
import matplotlib.pyplot as plt

"""
Code taken from https://github.com/Nik-Kras/ToMnet-N
@Author Nikita Krasnytskyi
@Modified by Filip Borowiak
"""

class World():
    def __init__(self, row_size, col_size, goal_rewards=None,
                 step_cost=-0.001, max_moves_per_episode=150,
                 consume_goals=1, shuffle=True, no_walls=False):

        self.width = row_size
        self.height = col_size
        
    # Set the reward for each goal A, B, C, D.
        # It could differ for each agent,
        # So, at the beginning of the game it sets for an agent individually
        self.action_space_size = 4
        self.shaffle = shuffle
        self.consumed_goal = "" 
        self.init_map = [" "] * self.width 
        self.players_position = [0, 0]
        

        if goal_rewards is None:
            goal_rewards = [2, 4, 8, 16]
        self.goal_rewards = goal_rewards

        self.consume_goals = consume_goals
        self.cnt_goal_picked = 0
        self.goal_picked = False

        # Set step cost in the environment
        # It could differ from experiment to experiment,
        # So, should be set at the beginning of the game

        self.step_cost = step_cost
        self.step_count = 0
        # Max number of moves after which player
        self.max_moves = max_moves_per_episode
        self.reward_punish = -20 # Previously was -1
        self.Generator = MazeGenerator(self.width, self.height, random_seed=42, max_rooms=15, no_walls=no_walls)
        self.state_matrix = [np.zeros((self.width, self.height), dtype=np.float16),
                             np.zeros((self.width, self.height), dtype=np.float16),
                             np.zeros((self.width, self.height), dtype=np.float16)] 
        
        self.MapEnum =[
            # self.PlayerMap
            {"Player": 1,
             "Other": 0},
            # self.WallMap
            {"Wall": 0,
             "Other": 1},
            # self.GoalMap
            {"Goal A": 1,
             "Goal B": 2,
             "Goal C": 3,
             "Goal D": 4,
             "Other": 0}
        ]

        self.objectsEnum = {
            "Wall": 0,
            "Path": 1,
            "Goal A": 2,
            "Goal B": 3,
            "Goal C": 4,
            "Goal D": 5,
            "Player": 10
            } 
        
        self.GoalValue = {
            self.objectsEnum["Goal A"]: 2,
            self.objectsEnum["Goal B"]: 4,
            self.objectsEnum["Goal C"]: 8,
            self.objectsEnum["Goal D"]: 16,
        }
        
        # Indexes for 3 map layers
        self.PlayerLayerIdx = 0
        self.WallLayerIdx = 1
        self.GoalLayerIdx = 2
        
# Shows specification on states data
    def states(self):
        # dict(type='int', shape=(self.world_row,self.world_col,), num_values=11)
        return dict(
            Player = dict(type='float', shape=(1, self.width, self.height)),
            Walls = dict(type='float',  shape=(1, self.width, self.height)),
            Goals = dict(type='float',  shape=(1, self.width, self.height))
        )

    # Shows specification on actions
    def actions(self):
        return dict(type='int', num_values=4)

    # Used for limiting episode with number of actions
    # Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return 90
    

    def reset(self):
        """Return initial_time_step."""

        #print("The RESET was called")

        # Clear the Map
        empty_map = np.zeros((3, self.width, self.height))
        for layer in range(3):
            self.state_matrix[layer] = empty_map
        self.consumed_goal=""
        self.cnt_goal_picked = 0
        self.goal_picked = False   
        # Create a new Map
        self.Generator.reset()
        self.walls_layout = self.Generator.getWallsLayout()
        self.goals_layout = self.Generator.getGoalsLayout()
        self.maze_layout = self.Generator.getMapLayout()

        self.maze_map_asArray = self.Generator.transformMazeToArray(self.maze_layout)


        self.state_matrix[self.WallLayerIdx] = self.Generator.CreateStateMatrix(self.walls_layout, "Wall")
        self.state_matrix[self.PlayerLayerIdx] = self.Generator.CreateStateMatrix(self.maze_layout, "Player")
        self.state_matrix[self.GoalLayerIdx] = self.Generator.CreateStateMatrix(self.goals_layout, "Goals")

        x, y = np.where(self.state_matrix[self.PlayerLayerIdx] == 1)
        self.players_position = [x[0], y[0]]
       
        # Clear step counter in the game
        self.step_count = 0

        # Set the initial map (for game saving)
        self.save_initial_map()

        dict_map = {"Player": np.expand_dims(self.state_matrix[self.PlayerLayerIdx], axis=0),
                    "Walls":  np.expand_dims(self.state_matrix[self.WallLayerIdx], axis=0),
                    "Goals":  np.expand_dims(self.state_matrix[self.GoalLayerIdx], axis=0)}

        # In the future, it should output Observed map (7x7), not "self.state_matrix"
        return dict_map
    
    def save_initial_map(self):
        for i in range(self.height):
            self.init_map[i] = '#'
            for j in range(self.width):
                if self.state_matrix[self.WallLayerIdx][i, j] == self.MapEnum[self.WallLayerIdx]["Wall"]:
                    self.init_map[i] = self.init_map[i] + '#'
                elif self.state_matrix[self.PlayerLayerIdx][ i, j] == self.MapEnum[self.PlayerLayerIdx]["Player"]:
                    self.init_map[i] = self.init_map[i] + 'O'
                elif self.state_matrix[self.GoalLayerIdx][ i, j] == self.MapEnum[self.GoalLayerIdx]["Goal A"]:
                    self.init_map[i] = self.init_map[i] + 'A'
                elif self.state_matrix[self.GoalLayerIdx][ i, j] == self.MapEnum[self.GoalLayerIdx]["Goal B"]:
                    self.init_map[i] = self.init_map[i] + 'B'
                elif self.state_matrix[self.GoalLayerIdx][ i, j] == self.MapEnum[self.GoalLayerIdx]["Goal C"]:
                    self.init_map[i] = self.init_map[i] + 'C'
                elif self.state_matrix[self.GoalLayerIdx][ i, j] == self.MapEnum[self.GoalLayerIdx]["Goal D"]:
                    self.init_map[i] = self.init_map[i] + 'D'
                else:
                    self.init_map[i] = self.init_map[i] + '-'
            self.init_map[i] = self.init_map[i] + '#'
        #print("Initial Map:")
        #print(self.init_map)

    def get_sight(self, sight, observability="partial"):
        self.simple_map = np.ones((self.width, self.height), dtype=np.int16)  # 0-wall, 1-path. Start with all path, then add walls, then add goals

        # Put walls
        for row in range(self.width):
            for col in range(self.height):
                if self.state_matrix[self.WallLayerIdx][row, col] == self.MapEnum[self.WallLayerIdx]["Wall"]: self.simple_map[row, col] = self.objectsEnum["Wall"]

        # Put goals
        for row in range(self.width):
            for col in range(self.height):
                if   self.state_matrix[self.GoalLayerIdx][row, col] == self.MapEnum[self.GoalLayerIdx]["Goal A"]: self.simple_map[row, col] = self.objectsEnum["Goal A"]
                elif self.state_matrix[self.GoalLayerIdx][row, col] == self.MapEnum[self.GoalLayerIdx]["Goal B"]: self.simple_map[row, col] = self.objectsEnum["Goal B"]
                elif self.state_matrix[self.GoalLayerIdx][row, col] == self.MapEnum[self.GoalLayerIdx]["Goal C"]: self.simple_map[row, col] = self.objectsEnum["Goal C"]
                elif self.state_matrix[self.GoalLayerIdx][row, col] == self.MapEnum[self.GoalLayerIdx]["Goal D"]: self.simple_map[row, col] = self.objectsEnum["Goal D"]

        if observability == "full":
            return self.players_position, self.simple_map

        result = np.full((sight, sight), None)

        half_sight = int(sight * 0.5)
        for i in range(sight):
            for j in range(sight):
                x = i + self.players_position[0] - half_sight
                y = j + self.players_position[1] - half_sight

                if -1 < x < self.width and -1 < y < self.height:
                    result[i, j] = self.simple_map[x, y]

        return self.players_position, result
    
    def check_terminate(self, action):

        # By default, everything is okay
        # If one of "bad" circumstances happens - it changes to True
        terminate = False
        goal_picked = 0

        # If player made more than max_moves (90) steps - terminate the game
        if self.step_count > self.max_moves: return [True, goal_picked]
        #elif self.step_count == 0: print("First Move!")
        self.step_count += 1

        # Check the boarders and
        # Move the player
        # Actions: 0 1 2 3 <-> UP RIGHT DOWN LEFT
        if   action == 0 and self.players_position[0] > 0: 
            new_position = [self.players_position[0] - 1, self.players_position[1]]
        elif action == 1 and self.players_position[1] < self.height - 1:
            new_position = [self.players_position[0], self.players_position[1] + 1]
        elif action == 2 and self.players_position[0] < self.width - 1:
            new_position = [self.players_position[0] + 1, self.players_position[1]]
        elif action == 3 and self.players_position[1] > 0:
            new_position = [self.players_position[0], self.players_position[1] - 1]
        else:
            #print("Player goes out of the borders")
            return [True, goal_picked] # This could be simplified to one big boolean expression instead of many if-else

        # Check if player has hit the wall on its move
        hit_wall = self.state_matrix[self.WallLayerIdx][new_position[0], new_position[1]] == self.MapEnum[self.WallLayerIdx]["Wall"]  #self.ObjSym["Wall"]
        if hit_wall: return [True, goal_picked]

        # Check if player picked a goal
        if   self.state_matrix[self.GoalLayerIdx][new_position[0], new_position[1]] == self.MapEnum[self.GoalLayerIdx]["Other"]:
            goal_picked = 0  # Path
        elif self.state_matrix[self.GoalLayerIdx][new_position[0], new_position[1]] == self.MapEnum[self.GoalLayerIdx]["Goal A"]:
            goal_picked = self.MapEnum[self.GoalLayerIdx]["Goal A"]
            self.consumed_goal = self.consumed_goal + "A"
            # terminate   = True # Picking two goals changes behaviour of terminate
        elif self.state_matrix[self.GoalLayerIdx][new_position[0], new_position[1]] == self.MapEnum[self.GoalLayerIdx]["Goal B"]:
            goal_picked = self.MapEnum[self.GoalLayerIdx]["Goal B"]
            self.consumed_goal = self.consumed_goal + "B"
            # terminate   = True # Picking two goals changes behaviour of terminate
        elif self.state_matrix[self.GoalLayerIdx][new_position[0], new_position[1]] == self.MapEnum[self.GoalLayerIdx]["Goal C"]:
            goal_picked = self.MapEnum[self.GoalLayerIdx]["Goal C"]
            self.consumed_goal = self.consumed_goal + "C"
            # terminate   = True # Picking two goals changes behaviour of terminate
        elif self.state_matrix[self.GoalLayerIdx][new_position[0], new_position[1]] == self.MapEnum[self.GoalLayerIdx]["Goal D"]:
            goal_picked = self.MapEnum[self.GoalLayerIdx]["Goal D"]
            self.consumed_goal = self.consumed_goal + "D"
            # terminate   = True # Picking two goals changes behaviour of terminate

        self.goal_picked = goal_picked
        return [terminate, goal_picked]
    


    ### ---- To correct ---- ###



    def check_reward(self, action, terminate, goal_picked):

        # By default, the reward is just a cost of step by ground
        # It will be changed to higher or lower rewards depending on new position
        reward = self.step_cost

        # If gone out of border or stepped on the wall
        if (terminate == True) and (goal_picked == False): return self.reward_punish

        # Acquiring new position
        new_position = [self.players_position[0], self.players_position[1]] # Initialize variable
        if   action == 0:  new_position = [self.players_position[0] - 1, self.players_position[1]]
        elif action == 1:  new_position = [self.players_position[0], self.players_position[1] + 1]
        elif action == 2 : new_position = [self.players_position[0] + 1, self.players_position[1]]
        elif action == 3 : new_position = [self.players_position[0], self.players_position[1] - 1]
        else: print("ERROR: The action is incorrect. Must be between 0 and 3, got: ", action)

        # Check receiving the goal in the next step and taking according reward
        if   self.state_matrix[self.GoalLayerIdx][new_position[0], new_position[1]] == self.MapEnum[self.GoalLayerIdx]["Other"]:  reward = self.step_cost        # Path
        elif self.state_matrix[self.GoalLayerIdx][new_position[0], new_position[1]] == self.MapEnum[self.GoalLayerIdx]["Goal A"]: reward = self.goal_rewards[0]  # Goal 1
        elif self.state_matrix[self.GoalLayerIdx][new_position[0], new_position[1]] == self.MapEnum[self.GoalLayerIdx]["Goal B"]: reward = self.goal_rewards[1]  # Goal 2
        elif self.state_matrix[self.GoalLayerIdx][new_position[0], new_position[1]] == self.MapEnum[self.GoalLayerIdx]["Goal C"]: reward = self.goal_rewards[2]  # Goal 3
        elif self.state_matrix[self.GoalLayerIdx][new_position[0], new_position[1]] == self.MapEnum[self.GoalLayerIdx]["Goal D"]: reward = self.goal_rewards[3]  # Goal 4
        else: print("ERROR: Incorrect map value! Position: ", new_position[0], ", ", new_position[1])

        return reward

    def move(self, action, terminate, goal_picked):

        # If gone out of border or stepped on the wall
        if (terminate == True) and (goal_picked == False): return self.state_matrix

        # Acquiring new position
        new_position = [self.players_position[0], self.players_position[1]]  # Initialize variable
        if   action == 0: new_position = [self.players_position[0] - 1, self.players_position[1]]
        elif action == 1: new_position = [self.players_position[0], self.players_position[1] + 1]
        elif action == 2: new_position = [self.players_position[0] + 1, self.players_position[1]]
        elif action == 3: new_position = [self.players_position[0], self.players_position[1] - 1]
        else: print("ERROR: The action is incorrect. Must be between 0 and 3, got: ", action)

        # Clear the current place of the player (COULD BE CHANGED WITH NEW ELEMENT TO SHOW TRAJECTORY)
        self.state_matrix[self.PlayerLayerIdx][self.players_position[0], self.players_position[1]] = self.MapEnum[self.PlayerLayerIdx]["Other"] # self.ObjSym["Path"]

        # Update the player's position!
        self.players_position = new_position
        self.state_matrix[self.PlayerLayerIdx][self.players_position[0], self.players_position[1]] = self.MapEnum[self.PlayerLayerIdx]["Player"] # self.ObjSym["Player"]

        dict_map = {"Player": np.expand_dims(self.state_matrix[self.PlayerLayerIdx], axis=0),
                    "Walls": np.expand_dims(self.state_matrix[self.WallLayerIdx], axis=0),
                    "Goals": np.expand_dims(self.state_matrix[self.GoalLayerIdx], axis=0)}

        return dict_map

    def execute(self, actions):

        action = actions
        
        if  0 >= action >= self.action_space_size:
            raise ValueError('The action is not included in the action space.')

        terminate, goal_picked = self.check_terminate(action)
        reward = self.check_reward(action, terminate, goal_picked)
        observe = self.move(action, terminate, goal_picked)

        # terminate = False # There is a new condition to terminate a game - pick two goals!

        # When the goal is picked - shaffle other goals
        if goal_picked != 0 and self.cnt_goal_picked == 0:
            self.cnt_goal_picked += 1

            if self.shaffle:
                # Delete picked goal
                self.delete_picked_goal()
                # Shaffle rest goals
                self.shuffle_goals()

        if goal_picked and self.cnt_goal_picked == self.consume_goals:
            terminate = True
        # if terminate: print("Episode is finished. Moves played: ", self.step_count, "Goal picked? ", goal_picked)
        return observe, terminate, goal_picked, reward

    """
            ################################  METHODS USED FOR TENSORFORCE  #####################################
    """

    def delete_picked_goal(self):
        self.state_matrix[self.GoalLayerIdx][self.players_position[0], self.players_position[1]] = self.MapEnum[self.GoalLayerIdx]["Other"]

    def shuffle_goals(self):

        # 1. Create dictionary of left goals and their coordinates
        leftgoals = dict()
        map = self.state_matrix[self.GoalLayerIdx]
        Goals = ["Goal A", "Goal B", "Goal C", "Goal D"]

        for row in range(self.width):
            for col in range(self.height):
                if map[row, col] != self.MapEnum[self.GoalLayerIdx]["Other"]:
                    index = int(map[row, col] - 1)
                    goal_name = Goals[index]
                    leftgoals[goal_name] = dict(position=[row, col], value=self.goal_rewards[index])

        # 2. Shaffle coordinates
        print("Goals before shuffle: ", leftgoals)
        # Random shaffle:
        # l = list(leftgoals.items())
        # random.shuffle(l)
        # leftgoals = dict(l)

        # Change least and most valuable goals:
        least_goal = dict(name="0", position=[0, 0], value=1000)
        most_goal  = dict(name="0", position=[0, 0], value=0)
        for key, leftgoal in leftgoals.items():
            if least_goal["value"] > leftgoal["value"]:
                least_goal["name"] = key # self.ValueGoal[leftgoal["value"]]
                least_goal["position"] = leftgoal["position"]
                least_goal["value"] = leftgoal["value"]
            if most_goal["value"] < leftgoal["value"]:
                most_goal["name"] = key # self.ValueGoal[leftgoal["value"]]
                most_goal["position"] = leftgoal["position"]
                most_goal["value"] = leftgoal["value"]

        print("The most valuable left goal: ", most_goal)
        print("The least valuable left goal: ", least_goal)

        new_leftgoals = dict()
        for key, leftgoal in leftgoals.items():
            if key == most_goal["name"]:
                leftgoal["position"] = least_goal["position"]
            elif key == least_goal["name"]:
                leftgoal["position"] = most_goal["position"]
            new_leftgoals[key] = leftgoal

        print("Goals after shuffle: ", new_leftgoals)

        # 3. Put new goals on the map
        map_new = np.zeros((self.width, self.height))
        for key, new_leftgoal in new_leftgoals.items():
            row = new_leftgoal["position"][0]
            col = new_leftgoal["position"][1]
            map_new[row, col] = self.MapEnum[self.GoalLayerIdx][key]


        #self.setStateMatrix(map_new, set="goals")

    """
        Clears all the map, preparing for a new one
    """
    def clear(self):
        self.reset()

    def getWorldState(self):
        return self.state_matrix

    def getPlayerPosition(self):
        return self.players_position

    def render(self):
        print(self.maze_layout)

    def draw_map(self):

        fig = plt.figure()
        #fig_grid = fig.add_subplot(111) # fig.add_subplot(121)
        #fig_health = fig.add_subplot(243)
        #fig_visible = fig.add_subplot(244)

        # fig.imshow(self.simple_map, vmin=-1, vmax=1, cmap='jet')
        #fig_visible.matshow(visible, vmin=-1, vmax=1, cmap='jet')
        # Render health chart
        #health_plot[i] = health
        # fig_health.clear()
        # fig_health.axis([0, frames, 0, 2])
        # fig_health.plot(health_plot[:i + 1])

        plt.axis('off')
        plt.imshow(self.maze_map_asArray)
        plt.show()
        print(self.maze_map_asArray)