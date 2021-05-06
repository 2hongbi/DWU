import pygame
import sys
import random


class SimpleGame:
    DISPLAY_SIZE = (640, 480)  # class variable

    def __init__(self, b_color='white'):
        pygame.init()

        self.screen = pygame.display.set_mode(SimpleGame.DISPLAY_SIZE)       # display
        pygame.display.set_caption('Generate Random Blocks')

        self.background = pygame.Surface(self.screen.get_size()).convert()     # background
        self.background.fill(b_color)

        self.block = pygame.Surface((10, 10)).convert()     # block
        self.block.fill(pygame.Color('red'))

        self.clock = pygame.time.Clock()        # Clock 객체

    def render(self, pos):
        '''
        :param pos: block의 위치 좌표
        :return: None
        '''
        b_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.block.fill(pygame.Color(b_color))
        self.screen.blit(self.background, (0,0))    # 스크린에 배경 배치
        self.background.blit(self.block, pos)           # 스크린에 블록을 배치
        # self.screen.blit(self.block, pos)           # 스크린에 블록을 배치

        pygame.display.flip()                       # 화면 업데이트

    def generate_block(self):
        self.clock.tick(3)

        # block_pos = (random.randint(0, 630), random.randint(0, 470))
        keep_going = True

        while keep_going:
            self.clock.tick(10)

            for event in pygame.event.get():        # 이벤트에 대한 처리
                if event.type == pygame.QUIT:
                    keep_going = False

            x = random.randint(0, self.screen.get_size()[0])
            y = random.randint(0, self.screen.get_size()[1])
            block_pos = (x, y)
            # block_pos = (random.randint(0, 620), random.randint(0, 460))
            self.render(block_pos)

        sys.exit()


if __name__ == '__main__':
    game = SimpleGame(pygame.Color('black'))
    game.generate_block()