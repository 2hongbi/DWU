import random


class Coin:
    def __init__(self):
        self.faceValue = random.randint(0, 1)

    def toss(self):
        self.faceValue = random.randint(0, 1)


class CoinGame:
    def __init__(self):
        self.coin1 = Coin()
        self.coin2 = Coin()


    def play(self):
        self.coin1.toss()
        fv1 = self.coin1.faceValue

        self.coin2.toss()
        fv2 = self.coin2.faceValue

        if fv1 == fv2:
            print(f'You Win! Both of Coin 1 and 2 are {"H" if fv1==0 else "T"}')
        else:
            print(f'You lose! Coin1 is {fv1} and Coin2 is {fv2}')


if __name__ == '__main__':
    game = CoinGame()
    game.play()