import datetime
# last_id = 0   # 전역 변수


class Note:
    last_id = 0  # class 소속 변수

    def __init__(self, memo):
        self.__memo = memo
        self.__creation_date = datetime.date.today()

        # global last_id
        Note.last_id += 1
        self.__id = Note.last_id

    @property
    def id(self):
        return self.__id

    @property
    def creation_date(self):
        return self.__creation_date

    @property
    def memo(self):
        return self.__memo

    @memo.setter
    def memo(self, txt):
        self.__memo = txt

    def match(self, txt):
        return txt in self.memo