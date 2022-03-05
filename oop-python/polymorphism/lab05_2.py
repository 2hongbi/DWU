from abc import abstractmethod, ABC


class Product(ABC):
    def __init__(self, price):
        self.__regularPrice = price

    @property
    def regularPrice(self):
        return self.__regularPrice

    @regularPrice.setter
    def regularPrice(self, new_price):
        self.__regularPrice = new_price

    def computeSalesPrice(self):
        return 0

    def __str__(self):
        return f'정상가 : {self.__regularPrice}'


class Book(Product):
    def __init__(self, price, pub, pub_yr, title):
        super().__init__(price)
        self.__publisher = pub
        self.__pub_year = pub_yr
        self.__title = title

    @property
    def publisher(self):
        return self.__publisher

    @property
    def pub_year(self):
        return self.__pub_year

    @property
    def title(self):
        return self.__title
        
    def computeSalesPrice(self):
        # return super().regularPrice * .5
        return self.regularPrice * .5

    def __str__(self):
        return super().__str__() + f' 출판사 : {self.publisher}, 출판년도 : {self.pub_year}, 제목 : {self.title}'


class ChildrenBook(Book):
    def __init__(self, price, pub, pub_yr, title, age):
        super().__init__(price, pub, pub_yr, title)
        self.__age = age

    @property
    def age(self):
        return self.__age

    def computeSalesPrice(self):
        # return super().regularPrice * 0.3
        return self.regularPrice

    def __str__(self):
        return super().__str__() + f' 대상 연령 : {self.__age}'


class Cartoon(Book):
    def __init__(self, price, pub, pub_yr, title, char_name):
        super().__init__(price, pub, pub_yr, title)
        self.__character_name = char_name

    @property
    def character_name(self):
        return self.__character_name

    def computeSalesPrice(self):
        # return super().regularPrice * 0.4
        return self.regularPrice

    def __str__(self):
        return super().__str__() + f' 캐릭터 : {self.character_name}'


if __name__ == '__main__':
    books = [Book(13800, '열린책들', 2013, '제3인류'),
             Book(38000, '창비', 2001, '한국현대사'),
             ChildrenBook(11700, '책읽는 곰', 2021, '어서 와! 장풍아', 7),
             Cartoon(21600, '시공사', 2021, '고담시티투어', '배트맨')]

    total_regPrice = 0
    total_salesPrice = 0

    print('구입 책 목록 : ')
    for book in books:
        print(book)
        total_regPrice += book.regularPrice
        total_salesPrice += book.computeSalesPrice()

    print(f'총 정상가 : {total_regPrice}')
    print(f'총 구입가 : {total_salesPrice}')
    print(f'할인액 : {total_regPrice - total_salesPrice}')