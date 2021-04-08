import random


class Coin:
    def __init__(self):
        self.__faceValue = random.randint(0, 1)

    def getFaceValue(self):
        if self.__faceValue == 0:
            return "H"
        else:
            return "T"
        # return self.__faceValue

    def toss(self):
        self.__faceValue = random.randint(0, 1)


class CoinGame:
    def __init__(self):
        self.coin1 = Coin()
        self.coin2 = Coin()

    def play(self):
        self.coin1.toss()
        fv1 = self.coin1.getFaceValue()

        self.coin2.toss()
        fv2 = self.coin2.getFaceValue()

        if fv1 == fv2:
            print(f'You Win! Both of Coin 1 and 2 are {fv1}')
        else:
            print(f'You lose! Coin1 is {fv1} and Coin2 is {fv2}')


if __name__ == '__main__':
    game = CoinGame()
    game.play()