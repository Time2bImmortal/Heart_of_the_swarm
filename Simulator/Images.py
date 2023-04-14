import time
import pygame
from pygame.locals import *
import random
from os import environ
import ArduinoSerial
import multiprocessing

'''Just adding the flip parameter for image direction and see SpriteObj to choose an image'''


class Animation(object):

    def __init__(self, screen, screen_size, win_fb, flip):
        # self.img = pygame.image.load("assets/spotlight.png")
        self.win_fb = win_fb
        self.flip = flip
        self.screen = screen
        self.screen_size = screen_size
        self.all_sprites_list = pygame.sprite.Group()
        for i in range(50):
            # stimuli = Ball(screen_size, screen_size, self.all_sprites_list)
            stimuli = SpriteObj(self.screen_size, self.win_fb, self.flip)
            self.all_sprites_list.add(stimuli)

    def run_logic(self):
        self.all_sprites_list.update()

    def display_frame(self):
        """ Display everything to the screen for the game. """
        self.screen.fill((255, 255, 255))
        self.all_sprites_list.draw(self.screen)
        pygame.display.update()


class OurWindow:

    def __init__(self, win_pos=(0, 0), state=0, win_fb=True, flip=True): # size=(1920,1080)
        self.y_origin, self.x_origin = win_pos
        self.state = state
        self.win_fb = win_fb
        self.flip =flip
        pygame.init()
        pygame.event.set_allowed([QUIT])

    def run_experiment(self, flag):

        self.initial_routine()
        self.animation = Animation(self.screen, self.screen.get_size(), win_fb=self.win_fb, flip=self.flip)
        self.pre_experiment()
        self.simple_loop(flag)

    def initial_routine(self):
        environ['SDL_VIDEO_WINDOW_POS'] = f"{self.y_origin},{self.x_origin}"
        self.screen = pygame.display.set_mode(flags=self.state)
        self.clock = pygame.time.Clock()

    def simple_loop(self, flag):
        start_time = time.time()
        x = 1
        counter = 0
        for i in range(1800):
            self.animation.run_logic()
        while 1:

            if flag.value:     #ICOB
                self.animation.display_frame()
                time.sleep(1)
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


class Block(pygame.sprite.Sprite):

    def __init__(self):
        super().__init__()
        # self.image = pygame.Surface([20, 20], pygame.SRCALPHA)
        # self.image.fill((255, 255, 255))
        self.image = pygame.image.load("images/NyLo 45x38.jpg")
        self.size = self.image.get_size()

        # create a 2x bigger image than self.image
        # self.smaller_img = pygame.transform.scale(self.image, (2,2))
        self.rect = self.image.get_rect()
        # draw bigger image to screen at x=100 y=100 position
        # self.screen.blit(self.smaller_img, [100, 100])
        # pygame.draw.circle(self.image, 'black', (10, 10), 10)
        # pygame.draw.polygon(self.image, 'black', [(0,0), (0,10), (20,0)])

    def reset_pos(self):
        """ Called when the block is 'collected' or falls off
            the screen. """
        # self.rect.x = random.randrange(0, WIDTH)
        self.rect.x = WIDTH
        self.rect.y = random.randrange(HEIGHT)

    def update(self):
        self.rect.x += -2
        if self.rect.x > WIDTH + self.rect.width or self.rect.x < 0:
            self.reset_pos()

    def pause(self):
        pass


class SpriteObj(pygame.sprite.Sprite):

    def __init__(self, screen_size, win_fb, flip):
        super().__init__()
        self.flip = flip
        self.width, self.height = screen_size
        self.win_fb = win_fb
        if flip:
            self.image = pygame.image.load("images/NyLo_180x180.jpg")
        else:
            self.image = pygame.image.load("images/NyLo_180x180flip.jpg")
        self.size = self.image.get_size()
        self.rect = self.image.get_rect()
        # pygame.draw.circle(self.image, (0, 0, 0), (22.5, 22.5), 22.5)

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

