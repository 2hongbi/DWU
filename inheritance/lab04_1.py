class BaseClass:
    def __init__(self):
        print('Creating an instance of Base Class')

    def myMethod(self):
        print('calling base method')


class Child(BaseClass):
    def __init__(self):
        super().__init__()
        print('Creating a child instance')

    def myMethod(self):
        print('calling child method')


if __name__ == '__main__':
    instance = Child()
    instance.myMethod()
