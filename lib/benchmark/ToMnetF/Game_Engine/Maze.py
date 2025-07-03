import numpy as np
import labmaze

"""
@Author Filip Borowiak
"""

class MazeGenerator():
    def __init__(self, width, height, random_seed, max_rooms, no_walls=False):
        self.world_row = width
        self.world_col = height
        self.random_seed = random_seed
        self.max_rooms = max_rooms
        self.maze_layout = ""
        self.walls_layout = ""
        self.goals_layout = ""
        self.player_layout = ""
        self.no_walls = no_walls
        # Indexes for 3 map layers
        self.PlayerLayerIdx = 0
        self.WallLayerIdx = 1
        self.GoalLayerIdx = 2

    
        self.objectsEnum = {
            "Wall": 0,
            "Goal A": 2,
            "Goal B": 3,
            "Goal C": 4,
            "Goal D": 5,
            "Player": 10
            } 
        
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
        
        if not no_walls:
            self.maze = labmaze.RandomMaze(height=self.world_col, width=self.world_row,
                                    random_seed=42, objects_per_room=1, max_rooms=max_rooms)
            
        else:
            MAZE_LAYOUT = '*************\n*           *\n*           *\n*           *\n*           *\n*           *\n*           *\n*           *\n*           *\n*           *\n*           *\n*           *\n*************\n'
            self.maze = labmaze.FixedMazeWithRandomGoals(MAZE_LAYOUT, num_spawns=1, num_objects=4)
            
        
    def reset(self):
        self.maze.regenerate()
        self.init()
        
    def init(self):
        maze_layout = str(self.maze._entity_layer)

        self.walls_layout = ""
        for s in maze_layout:
            self.walls_layout += s
        
        self.walls_layout = self.walls_layout.replace('G', " ", -1)
        
        
    
        maze = labmaze.FixedMazeWithRandomGoals(entity_layer=maze_layout,
                                               num_spawns=1, num_objects=4)
        self.maze_layout = ""
        for s in str(maze.entity_layer):
            self.maze_layout += s

        goals = ['1', '2', '3', '4']
        for goal in goals:
            self.maze_layout = self.maze_layout.replace('G', goal, 1)

        self.goals_layout = self.maze_layout.replace('P', ' ')
        self.goals_layout = self.goals_layout.replace('*', " ", -1)

    def getWallsLayout(self): return self.walls_layout

    def getGoalsLayout(self): return self.goals_layout

    def getMapLayout(self): return self.maze_layout

    def transformMazeToArray(self, dim_matrix):

        map =  np.ones((self.world_row, self.world_col),
                                  dtype=np.int16)
        dim_matrix = dim_matrix.replace('\n', '', -1)
        string_index = 0
        
        for row in range(self.world_row):
            for col in range(self.world_col):
                if dim_matrix[string_index] == '*':
                    map[row, col] = self.objectsEnum['Wall']
                    string_index += 1
                
                elif dim_matrix[string_index] == 'P':
                    map[row, col] = self.objectsEnum['Player']
                    string_index += 1

                elif dim_matrix[string_index] == '1':
                    map[row, col] = self.objectsEnum['Goal A']
                    string_index += 1
                
                elif dim_matrix[string_index] == '2':
                    map[row, col] = self.objectsEnum['Goal B']
                    string_index += 1
                
                elif dim_matrix[string_index] == '3':
                    map[row, col] = self.objectsEnum['Goal C']
                    string_index += 1

                elif dim_matrix[string_index] == '4':
                    map[row, col] = self.objectsEnum['Goal D']
                    string_index += 1
                    
                elif dim_matrix[string_index] == 'x':
                    map[row, col] = self.objectsEnum['Path']

                elif dim_matrix[string_index] == '\n':
                    string_index += 1
                else:
                    string_index += 1

        return map

    def CreateStateMatrix(self, dim_matrix, parameter):

        map =  np.ones((self.world_row, self.world_col),
                                  dtype=np.int16)
        dim_matrix = dim_matrix.replace('\n', '', -1)
        string_index = 0

        if parameter == 'Wall':
            for row in range(self.world_row):
                for col in range(self.world_col):
                    if dim_matrix[string_index] == '*':
                        map[row, col] = self.MapEnum[self.WallLayerIdx]['Wall']
                        string_index += 1
                    else:
                        string_index += 1
        
        elif parameter == 'Player':
            for row in range(self.world_row):
                for col in range(self.world_col):
                    if dim_matrix[string_index] == 'P':
                        map[row, col] = self.MapEnum[self.PlayerLayerIdx]['Player']
                        string_index += 1
                    else:
                        map[row, col] = self.MapEnum[self.PlayerLayerIdx]['Other']
                        string_index += 1

        
        elif parameter == 'Goals':
            for row in range(self.world_row):
                for col in range(self.world_col):
                    if dim_matrix[string_index] == '1':
                        map[row, col] = self.MapEnum[self.GoalLayerIdx]['Goal A']
                        string_index += 1
                    elif dim_matrix[string_index] == '2':
                        map[row, col] = self.MapEnum[self.GoalLayerIdx]['Goal B']
                        string_index += 1
                    elif dim_matrix[string_index] == '3':
                        map[row, col] = self.MapEnum[self.GoalLayerIdx]['Goal C']
                        string_index += 1
                    elif dim_matrix[string_index] == '4':
                        map[row, col] = self.MapEnum[self.GoalLayerIdx]['Goal D']
                        string_index += 1
                    else:
                        map[row, col] = self.MapEnum[self.GoalLayerIdx]['Other']
                        string_index += 1
        return map



        



    

        

    

