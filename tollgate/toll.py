import datetime
import random


class Car:
    def __init__(self, model, make, car_num, engine, seating, name, is_gov_vehicle=False):
        self.__model = model    # 자동차 모델 이름
        self.__make = make  # 생산년도
        self.__carNumber = car_num  # 번호판 이름
        self.__engineCapacity = engine  # 엔진 배기량
        self.__seatingCapacity = seating    # 최대 탑승인원
        self.__driverName = name    # 운전자 이름
        self.__isGovtVehicle = is_gov_vehicle   # 통행료 면제 대상 차량 여부

    @property
    def carNumber(self):
        return self.__carNumber

    def caculateToll(self):
        fee = 1000
        now_year = datetime.date.today().year

        if self.__isGovtVehicle:    # 통행료 면제대상 차량은 무료
            return 0

        # 배기량 계산
        if 1500 <= self.__engineCapacity < 2000:  # 1500cc 이상 ~2000cc 미만
            fee += self.__engineCapacity / 100
        elif 2000 <= self.__engineCapacity < 3000:    # 2000cc 이상 ~3000cc 미만
            fee += self.__engineCapacity / 65
        elif 3000 <= self.__engineCapacity:    # 3000cc 이상
            fee += self.__engineCapacity / 50

        # 최대 탑승가능 인원
        if self.__seatingCapacity >= 6:     # 6인승 이상
            fee += self.__seatingCapacity * 50

        # 생산년도 계산
        if now_year - self.__make + 1 >= 5:    # 5년이상 차량
            fee += (now_year - self.__make + 1) * 100

        return int(fee)

    def __str__(self):
        return f'자동차 모델: {self.__model}, 생산년도 : {self.__make}, 번호판 : {self.__carNumber}, ' \
               f'엔진 배기량 : {self.__engineCapacity}, 최대 탑승인원 : {self.__seatingCapacity}, ' \
               f'운전자 이름 : {self.__driverName}, 통행료 면제여부: {"면제" if self.__isGovtVehicle else "납부대상"}'


class Receipt:
    def __init__(self, car, timestamp):
        self.__car = car    # Car 객체 받아옴
        self.__timestamp = timestamp
        self.__tollPaid = self.__car.caculateToll()

    # def tollPaid(self):
    #     return self.__car.caculateToll()

    def __str__(self):
        return f'영수증 ---> 차 번호 : {self.__car.carNumber} / 금액 : {self.__tollPaid}원 / 시간 : {self.__timestamp}'


class Tollbooth:
    def __init__(self, location):
        self.__location = location
        self.__receipts = []

    @property
    def location(self):
        return self.__location

    @property
    def receipts(self):
        return self.__receipts

    def transit(self, car, time):   # car : Car 객체
        self.__receipts.append(Receipt(car, time))

    def __str__(self):
        result = f'톨 게이트 위치 : {self.__location}'
        for ret in self.__receipts:
            result += ret.__str__() + '\n'

        return result


class TollMngt:
    def __init__(self):
        self.__tollbooth = Tollbooth('성남')

    def add_cars(self, filename):
        car_file = open('./../data/'+filename, 'r', encoding='utf-16')
        car_list = []  # car 객체 받아옴

        for car_data in car_file:
            data = car_data.split(',')
            data = [item.strip() for item in data]  # data 리스트의 문자열 아이템의 앞뒤의 white space 삭제
            model, make, car_num, engine, seating, name, is_gov_vehicle = \
                    data[0], int(data[1]), data[2], int(data[3]), int(data[4]), data[5], eval(data[6])  # bool(1 if data[6] == 'True' else 0)
            car_list.append(Car(model, make, car_num, engine, seating, name, is_gov_vehicle))

        return car_list

    def run(self):
        cars = self.add_cars('cars.txt')    # cars 객체 반환 리스트
        timestamp = datetime.datetime.now()

        for car in cars:
            timestamp += datetime.timedelta(seconds=random.randint(10, 60))
            self.__tollbooth.transit(car, timestamp)    # car : Car 객체, timestamp

        print(f'톨부스 위치 : {self.__tollbooth.location}')
        for receipt in self.__tollbooth.receipts:
            print(receipt)

class 붕어빵:
    def __init__(self, 재료=팥):
        self.재료 = 재료


if __name__ == '__main__':
    tollmngt = TollMngt()
    tollmngt.run()




