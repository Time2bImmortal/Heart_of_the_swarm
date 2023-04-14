import time
import pygame
from pygame.locals import *
import random
from os import environ
import ArduinoSerial
import multiprocessing

'''For playing with the different control systems (open/closed loop), go to the simple loop function in OurWindow.'''

# SpriteObj class for creating a sprite with a fixed size and random position
class SpriteObj(pygame.sprite.Sprite):

    def __init__(self, screen_size, win_fb):
        super().__init__()
        self.width, self.height = screen_size
        self.win_fb = win_fb
        self.image = pygame.Surface([45, 45], pygame.SRCALPHA)
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()
        pygame.draw.circle(self.image, (0, 0, 0), (22.5, 22.5), 22.5)

    def reset_pos(self):
        self.rect.y = random.randrange(0, self.height)
        if self.win_fb:
            self.rect.x = random.randrange(self.width, self.width * 2)
        else:
            self.rect.x = random.randrange(self.width * -1, 0)

    def update(self):
        if self.win_fb:
            self.rect.x += -2
            if self.rect.x > self.width*2 or self.rect.x < 0:
                self.reset_pos()
        else:
            self.rect.x += 2
            if self.rect.x < self.width*-1 or self.rect.x > self.width:
                self.reset_pos()


# Animation class for managing the animation of sprites on the screen
class Animation(object):

    def __init__(self, screen, screen_size, win_fb):

        self.win_fb = win_fb
        self.screen = screen
        self.screen_size = screen_size
        self.all_sprites_list = pygame.sprite.Group()

        for i in range(60):
            stimuli = SpriteObj(self.screen_size, self.win_fb)
            self.all_sprites_list.add(stimuli)

    def run_logic(self):
        self.all_sprites_list.update()

    def display_frame(self):

        self.screen.fill((255, 255, 255))
        self.all_sprites_list.draw(self.screen)
        pygame.display.update()


# OurWindow class for managing the game window and running the experiment
class OurWindow:

    # Initialize the window with its position, state (fullscreen or not), and direction of the simulation (win_fb)
    def __init__(self, win_pos=(0, 0), state=0, win_fb=True): # size=(1920,1080)
        self.y_origin, self.x_origin = win_pos
        self.state = state
        self.win_fb = win_fb
        pygame.init()
        pygame.event.set_allowed([QUIT])

    # Function to run the experiment, including the initial routine, pre-experiment, and the main loop
    def run_experiment(self, flag):

        self.initial_routine()
        self.animation = Animation(self.screen, self.screen.get_size(), win_fb=self.win_fb)
        self.pre_experiment()
        self.simple_loop(flag)

    # Initial routine to set up the environment, screen, and clock for the game window
    def initial_routine(self):
        environ['SDL_VIDEO_WINDOW_POS'] = f"{self.y_origin},{self.x_origin}"
        self.screen = pygame.display.set_mode(flags=self.state)
        self.clock = pygame.time.Clock()

    # The main loop of the experiment, which continuously updates the animation and checks the flag value
    def simple_loop(self, flag):
        start_time = time.time()
        x = 1
        counter = 0
        for i in range(1800):
            self.animation.run_logic()
        while 1:
            '''Open-loop mode requires 'flag.value'. Shift to closed-loop mode by adding 'not'.'''
            if flag.value:
                self.animation.display_frame()
                time.sleep(1)  # When closed-loop 'with' comment this line to smooth the simulation.
                continue
            else:
                for event in pygame.event.get():
                    if event.type == 1:
                        pygame.quit()
                        exit()
                self.animation.run_logic()
                self.animation.display_frame()
                self.clock.tick(120)
            counter += 1
            if (time.time() - start_time) > x:
                print("FPS: ", int(counter / (time.time() - start_time)))
                counter = 0
                start_time = time.time()

    # Pre-experiment routine, which sets the screen color to white, black, and then white again, with pauses in between
    def pre_experiment(self):
        self.screen.fill((255, 255, 255))
        pygame.display.update(self.screen.get_rect())
        time.sleep(20)
        self.screen.fill((0, 0, 0))
        pygame.display.update(self.screen.get_rect())
        time.sleep(20)
        self.screen.fill((255, 255, 255))
        pygame.display.update(self.screen.get_rect())
        time.sleep(20)

