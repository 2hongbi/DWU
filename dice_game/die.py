import random


class Die:
    def __init__(self):
        self.__feceValue = random.randint(1, 6)

    def getFaceValue(self):
        return self.__faceValue

    def roll(self):
        self.__faceValue = random.randint(1, 6)


class DiceGame:
    def __init__(self):
        self.die1 = Die()
        self.die2 = Die()

    def play(self):
        self.die1.roll()
        fv1 = self.die1.getFaceValue()

        self.die2.roll()
        fv2 = self.die2.getFaceValue()

        if fv1+fv2 == 7:
            print('You win!')
        else:
            print(f'You lose! Sum of dice: {fv1+fv2}')