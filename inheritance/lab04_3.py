from abc import abstractmethod, ABC


class Employee(ABC):
    def __init__(self, name, id):
        self.__name = name
        self.__emp_id = id

    @property
    def name(self):
        return self.__name

    @property
    def emp_id(self):
        return self.__emp_id

    @abstractmethod
    def gross_pay(self):
        pass

    def __str__(self):
        return f'이름: {self.__name}, 아이디 : {self.__emp_id}'

class FullTime_Employee(Employee):
    def __init__(self, name, id, salary, bonus=0):
        super().__init__(name, id)
        self.__salary = salary
        self.__bonus = bonus

    @property
    def salary(self):
        return self.__salary

    @property
    def bonus(self):
        return self.__bonus

    @salary.setter
    def salary(self, new_salary):
        self.__salary = new_salary

    @bonus.setter
    def bonus(self, new_bonus):
        self.__bonus = new_bonus

    def gross_pay(self):
        return self.__salary + self.__bonus

    def __str__(self):
        att = super().__str__()
        att += f'월급여 : {self.__salary}, 보너스 : {self.__bonus}'
        return att


class PartTime_Employee(Employee):
    def __init__(self, name, id, h_wage, h_worked):
        super().__init__(name, id)
        self.__hourly_wage = h_wage
        self.__hourly_worked = h_worked

    @property
    def hourly_wage(self):
        return self.__hourly_wage

    @property
    def hourly_worked(self):
        return self.__hourly_worked

    @hourly_wage.setter
    def hourly_wage(self, new_wage):
        self.__hourly_wage = new_wage

    @hourly_worked.setter
    def hourly_worked(self, new_time):
        self.__hourly_worked = new_time

    def gross_pay(self):
        return self.__hourly_wage + self.__hourly_worked

    def __str__(self):
        return super().__str__() + f'시급 : {self.__hourly_wage}, 근무시간 : {self.__hourly_worked}'


if __name__ == '__main__':
    emps = [PartTime_Employee('이주원', '20180101', 9000, 42),
            PartTime_Employee('김하준', '20170101', 10000, 48),
            FullTime_Employee('박민준', '20140301', 3000000),
            FullTime_Employee('김민서', '20131201', 3600000, 1000000)]

    for emp in emps:
        print(emp)
        print('급여 : ', emp.gross_pay())