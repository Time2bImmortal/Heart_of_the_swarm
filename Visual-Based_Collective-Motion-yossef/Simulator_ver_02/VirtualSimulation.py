import time
import pygame
from pygame.locals import *
import random
from os import environ


# Class for creating a window structure for the simulation
class WindowStructure:
    def __init__(self, win_pos=(0, 0), state=0):
        self.y_origin, self.x_origin = win_pos
        self.state = state

    # Method for creating a window with a given position and flags
    def create_window(self):
        environ['SDL_VIDEO_WINDOW_POS'] = f"{self.y_origin},{self.x_origin}"
        self.screen = pygame.display.set_mode(flags=self.state)
        self.clock = pygame.time.Clock()


# Class for running the simulation
class Simulation(WindowStructure):
    def __init__(self, win_pos, control_system, state, num_sprites, direction, velocity, shift_direction,
                 shift_time, black_screen_duration, trial_duration):
        super().__init__(win_pos, state)
        self.create_window()
        self.num_sprites = num_sprites
        self.direction = direction
        self.velocity = velocity
        self.shift_direction = shift_direction
        self.shift_time = shift_time
        self.control_system = control_system
        self.black_screen_duration = black_screen_duration
        self.trial_duration = trial_duration

    # Method to run the experiment
    def run_experiment(self, flag):

        pygame.event.set_allowed([QUIT])
        self.pre_experiment()
        self.simple_loop_direction(flag)
        self.display_black_screen()
        if self.direction == "forward":
            self.direction = "backward"
        else:
            self.direction = "forward"
        self.pre_experiment()
        self.simple_loop_direction(flag)   # Need to insert a flag to arrest the whole program
        print(" -------- END OF THE SIMULATION --------")
        # exit(0)

    def display_black_screen(self):
        self.screen.fill((0, 0, 0))
        pygame.display.update()
        time.sleep(self.black_screen_duration)

    # Method to create an animation with given parameters
    def create_animation(self):
        all_sprites_list = pygame.sprite.Group()
        for _ in range(self.num_sprites):
            stimuli = SpriteObj(self.screen.get_size(), self.direction, self.velocity)
            all_sprites_list.add(stimuli)
        return Animation(self.screen, self.screen.get_size(), self.num_sprites, self.direction, self.velocity,
                         self.shift_direction)

    # Main loop for running the simulation
    def simple_loop_pause(self, flag):
        start_time = time.time()
        counter = 0
        shift_counter = 0
        animation = self.create_animation()
        trial_duration_counter = time.time()
        while time.time() - trial_duration_counter < self.trial_duration:
            shift_counter += 1
            if animation.shift_direction and shift_counter > self.shift_time * 120:
                animation.change_direction()
                shift_counter = 0
            if self.control_system == 0 or self.control_system == 2:
                if flag.value:
                    animation.display_frame()
                    time.sleep(1)
                    continue
            else:
                if not flag.value:
                    animation.display_frame()
                    continue
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()

            animation.run_logic()
            animation.display_frame()
            self.clock.tick(120)
            counter += 1
            if (time.time() - start_time) > 1:
                print("FPS: ", int(counter / (time.time() - start_time)))
                counter = 0
                start_time = time.time()

    def simple_loop_direction(self, flag):
        start_time = time.time()
        trial_duration_counter = time.time()
        counter = 0
        shift_counter = 0
        animation = self.create_animation()
        prev_flag_value = flag.value
        while time.time() - trial_duration_counter < self.trial_duration:
            shift_counter += 1
            if animation.shift_direction and shift_counter > self.shift_time * 120:
                animation.change_direction()
                shift_counter = 0

            if prev_flag_value != flag.value:
                if self.direction == "forward":
                    self.direction = "backward"
                else:
                    self.direction = "forward"

                for sprite in animation.all_sprites_list:
                    sprite.direction = self.direction

                prev_flag_value = flag.value

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    exit()

            animation.run_logic()
            animation.display_frame()
            self.clock.tick(120)
            counter += 1
            if (time.time() - start_time) > 1:
                print("FPS: ", int(counter / (time.time() - start_time)))
                counter = 0
                start_time = time.time()

    # Method to perform pre-experiment tasks
    def pre_experiment(self):  # 20 white, black, white
        self.screen.fill((255, 255, 255))
        pygame.display.update(self.screen.get_rect())
        time.sleep(20)
        self.screen.fill((0, 0, 0))
        pygame.display.update(self.screen.get_rect())
        time.sleep(20)
        self.screen.fill((255, 255, 255))
        pygame.display.update(self.screen.get_rect())
        time.sleep(20)


# Class for creating sprite objects
class SpriteObj(pygame.sprite.Sprite):

    def __init__(self, screen_size, direction, velocity, image=None):
        super().__init__()
        self.width, self.height = screen_size
        self.direction = direction
        self.velocity = velocity
        self.image_data = None
        self.create_surface(image)
        self.reset_pos()

    # Method to create a sprite surface with an image or default data
    def create_surface(self, image=None):
        if image is None:
            self.image_data = (45, 45)
        else:
            self.image_data = image
        self.generate_surface()

    # Method to generate the sprite surface using image_data
    def generate_surface(self):
        if self.image_data is None:
            width, height = 45, 45
        else:
            width, height = self.image_data
        self.image = pygame.Surface([width, height], pygame.SRCALPHA)
        self.image.fill((255, 255, 255))
        pygame.draw.circle(self.image, (0, 0, 0), (22.5, 22.5), 22.5)
        self.rect = self.image.get_rect()

    # Method to reset the position of the sprite
    def reset_pos(self):
        self.rect.y = random.randrange(0, self.height)
        if self.direction == "forward":
            self.rect.x = random.randrange(self.width, self.width * 2)
        else:
            self.rect.x = random.randrange(self.width * -1, 0)

    # Method to update the sprite's position based on its direction and velocity
    def update(self):
        if self.direction == "forward":
            self.rect.x += -self.velocity
            if self.rect.x > self.width * 2 or self.rect.x < 0:
                self.reset_pos()
        else:
            self.rect.x += self.velocity
            if self.rect.x < self.width * -1 or self.rect.x > self.width:
                self.reset_pos()


class Animation(object):

    def __init__(self, screen, screen_size, num_sprites, direction, velocity, shift_direction):
        self.direction = direction
        self.screen = screen
        self.num_sprites = num_sprites
        self.velocity = velocity
        self.screen_size = screen_size
        self.shift_direction = shift_direction
        self.all_sprites_list = pygame.sprite.Group()
        self.create_sprites()

    # Method to create sprite objects and add them to the sprite group
    def create_sprites(self):
        screen_size = self.screen.get_size()
        for _ in range(self.num_sprites):
            stimuli = SpriteObj(screen_size, self.direction, self.velocity)
            self.all_sprites_list.add(stimuli)

    # Method to change the direction of all sprites in the animation
    def change_direction(self):
        for sprite in self.all_sprites_list:
            if sprite.direction == "forward":
                sprite.direction = "backward"
            else:
                sprite.direction = "forward"

    # Method to display the current frame of the animation
    def display_frame(self):
        self.screen.fill((255, 255, 255))
        self.all_sprites_list.draw(self.screen)
        pygame.display.update()

    # Method to update the logic of the animation, including the position of the sprites
    def run_logic(self):
        self.all_sprites_list.update()


