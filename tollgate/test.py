import random
from datetime import datetime, timedelta

class Car:
    def __init__(self, model, make, carNumber, engineCapacity, seatingCapacity, driverName, isGovtVehicle):
        self.__model = model
        self.__make = make
        self.__carNumber = carNumber
        self.__engineCapacity = engineCapacity
        self.__seatingCapacity = seatingCapacity
        self.__driverName = driverName
        self.__isGovtVehicle = isGovtVehicle

    @property
    def model(self):
        return self.__model

    @property
    def make(self):
        return self.__make

    @property
    def carNumber(self):
        return self.__carNumber

    @carNumber.setter
    def carNumber(self, Number):
        self.__carNumber = Number

    @property
    def engineCapacity(self):
        return self.__engineCapacity

    @property
    def driverName(self):
        return self.__driverName

    @driverName.setter
    def driverName(self, name):
        self.__driverName = name

    @property
    def isGovtVehicle(self):
        return self.__isGovtVehicle

    @isGovtVehicle.setter
    def isGovtVehicle(self, x):
        self.__isGovtVehicle = x


    def calculateToll(self):
        charge = 1000
        if self.__isGovtVehicle == True:
            charge = 0
            return charge
        else:
            if self.__engineCapacity < 1500:
                charge = charge
            elif self.__engineCapacity >= 1500 | self.__engineCapacity < 2000:
                charge += self.__engineCapacity / 100
            elif self.__engineCapacity >= 2000 | self.__engineCapacity < 3000:
                charge += self.__engineCapacity / 65
            else:
                charge += self.__engineCapacity / 50
            if self.__seatingCapacity <= 5:
                charge = charge
            else:
                charge += self.__seatingCapacity * 50
            if datetime.today().year - self.__make < 5:
                charge = charge
            else:
                charge += (datetime.today().year - self.__make) * 100

            return int(charge)


class Tollbooth:
    def __init__(self, location):  #Tollbooth 생성자
        self.__location = location
        self.__receipts = []      #영수증 객체 리스트

    @property
    def location(self):
        return self.__location

    @property
    def receipts(self):
        return self.__receipts

    # transit에 구현해야 할 것
    # car = Car(그랜저 HG240, 2014, 29가1234, 2400, 5, 김길동, False)
    # tollbooth.receipts.append(Receipt(car, datetime.datetime.now()))
    # print(tollbooth)

    def transit(self, car, timestamp): #이거 구현해야됨 tollbooth의 receipts객체에 receipt 집어넣으면 됨
        self.__receipts.append(Receipt(car, timestamp))


    def __str__(self):
        t = f'톨게이트 위치: {self.__location}\n'
        for rct in self.__receipts:
            t += rct.__str__() + '\n'
        return t


class Receipt:
    def __init__(self, car, timestamp):
        self.__car = car
        self.__timestamp = timestamp
        self.tollPaid = self.car.calculateToll()

    @property
    def car(self):
        return self.__car

    @property
    def timestamp(self):
        return self.__timestamp

    def __str__(self):
        t = f'자동차 번호: {self.__car.carNumber}, 통과시간:{self.__timestamp}, 통행료: {self.tollPaid}'
        return t


class TollMngt:
    def __init__(self):
        self.tollbooth = Tollbooth('성남')

    def add_cars(self, filename):
        cars = []
        car_file = open('../data/cars.txt', 'r', encoding='utf-16')
        for car_data in car_file:
            data = car_data.split(',')
            data = [itm.strip() for itm in data]
            model, make, num, engine, seats, driver, free = data
            make, engine, seats, free = int(make), int(engine), int(seats), eval(free)
            car = Car(model, make, num, engine, seats, driver, free)
            cars.append(car)

        return cars


    def run(self):
        cars = self.add_cars('../data/cars.txt')
        timestamp = datetime.now()

        for car in cars:
            self.tollbooth.transit(car, timestamp)
            timestamp += timedelta(seconds=random.randint(10, 60))

        print(f'톨부스 위치: {self.tollbooth.location}\n')
        for receipt in self.tollbooth.receipts:
            print(receipt)


if __name__ == '__main__':
    tollmngt = TollMngt()
    tollmngt.run()