from abc import abstractmethod


class Car:
    def __init__(self, name):
        self.__name = name

    @property
    def name(self):
        return self.__name

    @abstractmethod
    def drive(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def __str__(self):
        return f'모델명 : {self.name}'


class SprotsCar(Car):
    def __init__(self, name):
        super().__init__(name)

    def drive(self):
        return 'Driving Sports Car.'

    def stop(self):
        return 'Breaking Sports Car.'


class Truck(Car):
    def drive(self):
        return 'Driving Truck.'

    def stop(self):
        return 'Breaking Truck.'


if __name__ == '__main__':
    cars = []
    cars.append(Truck('Banana Truck'))
    cars.append(SprotsCar('Z3'))

    for car in cars:
        print(car)
        print(car.drive())      # polymorphism 구현