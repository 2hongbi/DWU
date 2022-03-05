from datetime import date


class Pet:
    def __init__(self, name, birth, species):   # birth : '2020-01-01'
        self._name = name       # protected attribute
        self._birthday = date.fromisoformat(birth)
        self._species = species

    @property
    def name(self):
        return self._name

    @property
    def birthday(self):
        return self._birthday

    @property
    def species(self):
        return self._species

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def age(self):
        return date.today().year - self._birthday.year

    def eat(self):
        print('Yummy yum yum')

    def sleep(self):
        print('Zzzzzzz...')

    def __str__(self):
        return f'이름 : {self.name}, 종 : {self.species}, 나이 : {self.age}'


class Dog(Pet):
    def __init__(self, name, birth, species):
        Pet.__init__(self, name, birth, species)

    def bark(self):
        print('Bark! bark!')


class Cat(Pet):
    def __init__(self, name, birth, species):
        super().__init__(name, birth, species)

    def mew(self):
        print('Mew! mew!')


if __name__ == '__main__':
    myPets = [Dog('Domi', '2020-01-01', 'Maltese'),
              Cat('Nyangi', '2018-04-02', 'Scottish Fold')]

    for pet in myPets:
        if isinstance(pet, Dog):
            print('Pet Dog : ', end='\t')
            pet.bark()
        elif isinstance(pet, Cat):
            print('Pet Cat : ', end='\t')
            pet.mew()

        pet.eat()
        pet.sleep()
