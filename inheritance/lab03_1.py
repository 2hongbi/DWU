class BaseClass:
    def __init__(self, a):
        self.__a = a

    @property
    def a(self):
        return self.__a


class MyClass1(BaseClass):
    def __init__(self, x, b):
        super().__init__(x)
        self.__b = b

    @property
    def b(self):
        return self.__b


class MyClass2(BaseClass):
    def __init__(self, x, c):
        BaseClass.__init__(self, x)   # self.x 가 아닌 self, x
        self.__c = c

    @property
    def c(self):
        return self.__c


if __name__ == '__main__':
    cla1 = MyClass1(10, 20)
    cla2 = MyClass2(11, 100)

    print(f'상속 상속 값 : {cla1.a}, {cla1.b}')
    print(f'상속 상속 값 : {cla2.a}, {cla2.c}')

    print(issubclass(MyClass1, BaseClass))  # True
    print(issubclass(MyClass1, MyClass2))   # False

    print(isinstance(cla1, MyClass1))   # True
    print(isinstance(cla2, MyClass1))   # False
