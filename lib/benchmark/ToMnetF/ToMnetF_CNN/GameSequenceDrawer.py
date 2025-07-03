
import numpy as np
import pygame
import cv2

"""
@Author Filip Borowiak
"""

def drawMap(WIDTH, HEIGHT, simple_map, player_position, init=True, predictions=None, actual=None):

  STEP = 30

  pygame.init()
  screen = pygame.display.set_mode(size=(STEP*(WIDTH+2), STEP*(HEIGHT+2)))
  done = False
  is_blue = True
  x = 30
  y = 30

  # color of background
  color = (51, 51, 51)
    
  # Changing surface color
  screen.fill(color)

  # color of path
  # color = (255, 250, 205) # LemonChiffon
  color = (255, 250, 225)

  #print("simple_map:\n ", simple_map)
  #print("simple_map[0][0]: ", simple_map[0][0])

  # Create path
  for i in range(WIDTH):
    for j in range(HEIGHT):
      if simple_map[i][j] != 0:
        pygame.draw.rect(screen, color, pygame.Rect((i+1)*STEP, (j+1)*STEP, 
                                                    STEP, STEP))
        
  color = (0, 0, 0)      
  # Create borders
  pygame.draw.rect(screen, color, pygame.Rect(0, 0, STEP, STEP*(WIDTH+2)))
  pygame.draw.rect(screen, color, pygame.Rect(0, 0, STEP*(HEIGHT+2), STEP))
  pygame.draw.rect(screen, color, pygame.Rect(0, STEP*(HEIGHT+1), STEP*(WIDTH+2), STEP))
  pygame.draw.rect(screen, color, pygame.Rect(STEP*(WIDTH+1), 0, STEP, STEP*(WIDTH+2)))

  color = (255,0,255)
 
  # Create player
  if init:
    for i in range(WIDTH):
        for j in range(HEIGHT):
            if simple_map[i][j] == 10:
                pygame.draw.circle(screen, color, center=((i+1)*STEP+STEP/2, (j+1)*STEP+STEP/2), 
                            radius = STEP/2)

                posX, posY = [i, j]
                if actual is not None:
                   salmon = (255, 211, 210)
                   for cord in actual:

                        pygame.draw.rect(screen, salmon, pygame.Rect((cord[0]+1)*STEP, (cord[1]+1)*STEP,STEP, STEP))

                if predictions is not None:
                    yellow = (203, 217, 15)
                    for cord in predictions:
                        pygame.draw.rect(screen, yellow, pygame.Rect((cord[0]+1)*STEP, (cord[1]+1)*STEP,STEP, STEP))


     
  else:
    posX, posY = player_position
    if actual is not None:
                   salmon = (255, 211, 210)
                   for cord in actual:
                        pygame.draw.rect(screen, salmon, pygame.Rect((cord[0]+1)*STEP, (cord[1]+1)*STEP,STEP, STEP))
    pygame.draw.circle(screen, color, center=((posX+1)*STEP+STEP/2, (posY+1)*STEP+STEP/2), 
                            radius = STEP/2)
    
            
  #q = STEP/3
  q = 0
  r = 0
  # Create goals
  for i in range(WIDTH):
    for j in range(HEIGHT):
      if simple_map[i][j] == 2:
        #color = (255,0,0)  # Red
        color = (230,0,0)
        pygame.draw.rect(screen, color, pygame.Rect((i+1)*STEP + q/2, (j+1)*STEP + q/2, 
                                                    STEP - q, STEP - q), border_radius=r)
      elif simple_map[i][j] == 3:
        #color = (0,255,0)  # Green
        color = (0,230,0)
        pygame.draw.rect(screen, color, pygame.Rect((i+1)*STEP + q/2, (j+1)*STEP + q/2, 
                                                    STEP - q, STEP - q), border_radius=r)
        
      elif simple_map[i][j] == 4:
        #color = (0,255,255) # Blue
        color = (0,0,230) # Blue
        pygame.draw.rect(screen, color, pygame.Rect((i+1)*STEP + q/2, (j+1)*STEP + q/2, 
                                                    STEP - q, STEP - q), border_radius=r)
        
      elif simple_map[i][j] == 5:
        # color = (0,0,255)  # Cyan
        color = (0,230,230)
        pygame.draw.rect(screen, color, pygame.Rect((i+1)*STEP + q/2, (j+1)*STEP + q/2, 
                                                    STEP - q, STEP - q), border_radius=r)

 
  view = pygame.surfarray.array3d(screen)
  #  convert from (width, height, channel) to (height, width, channel)
  view = view.transpose([1, 0, 2])
  #  convert from rgb to bgr
  img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
  #Display image, clear cell every 0.5 seconds
  cv2.imshow("window1", img_bgr)
  cv2.waitKey(0)
  return posX, posY

simple_map = map
W=13
H=13

# actions - list of actions per game
# simple_map - converted map to ndarray

#Actions = {"Up": 0,
 #          "Right": 1,
  #         "Down": 2,
   #        "Left": 3}

def newCords(player_position, action):
    posX, posY = player_position
    if action == 0:
        player_pos = [posX, posY-1]
    elif action == 1:
        player_pos = [posX + 1, posY]
    elif action == 2:
        player_pos = [posX, posY+1]
    elif action == 3:
        player_pos = [posX - 1 , posY]
         
    return player_pos 



