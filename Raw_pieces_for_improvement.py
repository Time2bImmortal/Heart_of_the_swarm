# Ball class for creating a ball sprite with random color and position

# class Ball(pygame.sprite.Sprite):
#
#     def __init__(self, pos, screen_size, *groups):
#         super().__init__(groups)
#         self.image = pygame.Surface((30, 30), pygame.SRCALPHA)
#         self.screen_size = screen_size
#         col = random.randrange(256), random.randrange(256), random.randrange(256)
#         pygame.draw.circle(self.image, col, (15, 15), 15)
#         self.rect = self.image.get_rect(center=pos)
#         self.vel = pygame.math.Vector2(8, 0).rotate(random.randrange(360))
#         self.pos = pygame.math.Vector2(pos)
#
#     def update(self):
#         self.pos += self.vel
#         self.rect.center = self.pos
#         if self.rect.left < 0 or self.rect.right > WIDTH:
#             self.vel.x *= -1
#         if self.rect.top < 0 or self.rect.bottom > HEIGHT:
#             self.vel.y *= -1


# NEED TO MOVE STIMULIS WITH VECTORS, LIKE THAT I CAN ROATE, ACCELERATE THEM, MORE EASILY, AND NEED TO CODE
# A jump method
# have to add something to load images

# self.image= pygame.image.load(picture_path)

# ------------------------------------------
# some code for jumping
# while run:
#
#     # completely fill the surface object
#     # with black colour
#     win.fill((0, 0, 0))
#
#     # drawing object on screen which is rectangle here
#     pygame.draw.rect(win, (255, 0, 0), (x, y, width, height))
#
#     # iterate over the list of Event objects
#     # that was returned by pygame.event.get() method.
#     for event in pygame.event.get():
#
#         # if event object type is QUIT
#         # then quitting the pygame
#         # and program both.
#         if event.type == pygame.QUIT:
#             # it will make exit the while loop
#             run = False
#     # stores keys pressed
#     keys = pygame.key.get_pressed()
#
#     if isjump == False:
#
#         # if space bar is pressed
#         if keys[pygame.K_SPACE]:
#             # make isjump equal to True
#             isjump = True
#
#     if isjump:
#         # calculate force (F). F = 1 / 2 * mass * velocity ^ 2.
#         F = (1 / 2) * m * (v ** 2)
#
#         # change in the y co-ordinate
#         y -= F
#
#         # decreasing velocity while going up and become negative while coming down
#         v = v - 1
#
#         # object reached its maximum height
#         if v < 0:
#             # negative sign is added to counter negative velocity
#             m = -1
#
#         # objected reaches its original state
#         if v == -6:
#             # making isjump equal to false
#             isjump = False
#
#             # setting original values to v and m
#             v = 5
#             m = 1
#
#     # creates time delay of 10ms
#     pygame.time.delay(10)

# class Basic_sprite(pygame.sprite.Sprite):
#     """ This class represents a simple block the player collects. """
#
#     def __init__(self):
#         """ Constructor, create the image of the block. """
#         super().__init__()
#         self.image = pygame.Surface([20, 20])
#         self.image.fill((255, 255, 255))
#         self.rect = self.image.get_rect()
#
#     def reset_pos(self):
#         """ Called when the block is 'collected' or falls off
#             the screen. """
#         self.rect.x = random.randrange(0, WIDTH)
#         self.rect.y = random.randrange(HEIGHT)
#
#     def update(self):
#         """ Automatically called when we need to move the block. """
#         self.rect.x += -2
#
#         if self.rect.x > WIDTH + self.rect.width or self.rect.x < 0:
#             self.reset_pos()

#--------------------------
# colours = {'black': (0, 0, 0), 'white': (255, 255, 255)}
# Sprites_appearance = {}

