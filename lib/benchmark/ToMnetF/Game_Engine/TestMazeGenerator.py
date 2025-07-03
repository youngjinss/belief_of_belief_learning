from Maze import MazeGenerator
import matplotlib.pyplot as plt
import numpy as np

"""
@Author Filip Borowiak
"""
width = 25
height = 25
seed = 42
mg = MazeGenerator(width=width, height=height, random_seed=seed, max_rooms=20)

def run():
    
    walls = mg.getWallsLayout()
    goals = mg.getGoalsLayout()
    maze = mg.getMapLayout()


    maze_map = mg.transformMazeToArray(maze)

    state_goal_matrix = mg.CreateStateMatrix(goals, "Goals")

    state_wall_matrix = mg.CreateStateMatrix(walls, "Wall")

    state_player_matrix = mg.CreateStateMatrix(maze, "Player")


    x, y = np.where(state_player_matrix == 1)
    x = x[0]
    y = y[0]


    fig = plt.figure()
    plt.axis('off')
    plt.title("wall map!")
    plt.imshow(state_wall_matrix)
    plt.show()

    fig = plt.figure()
    plt.axis('off')
    plt.title("goal map!")
    plt.imshow(state_goal_matrix)
    plt.show()

    fig = plt.figure()
    plt.axis('off')
    plt.title("player map!")
    plt.imshow(state_player_matrix)
    plt.show()


    fig = plt.figure()
    plt.axis('off')
    plt.title("Maze map!")
    plt.imshow(maze_map)
    plt.show()


mg.init()
run()
#mg.reset()
#run()
#mg.reset()
#run()