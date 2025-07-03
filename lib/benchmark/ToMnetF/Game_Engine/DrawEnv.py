from Env import World
"""
@Author Filip Borowiak
"""

W = 25
H = 25

world = World(W, H, consume_goals=1, shuffle=False)

for _ in range(2):
    world.reset()
    world.render()
    world.draw_map()
