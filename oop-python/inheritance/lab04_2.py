from abc import abstractmethod, ABC


class MyClass1(ABC):
    def __init__(self):
        self.a = 1

    @abstractmethod
    def do_something(self):
        pass


class MyClass2(MyClass1):
    def __init__(self):
        super().__init__()
        self.b = 10

    def do_something(self):
        return self.a + self.b


if __name__ == '__main__':
    # inst_a = MyClass1() # Myclass1은 추상클래스로 객체 생성 X
    inst_b = MyClass2()

    # print(inst_a.do_something())
    print(inst_b.do_something())