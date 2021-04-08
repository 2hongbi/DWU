import datetime
# last_id = 0   # 전역 변수


class Note:
    last_id = 0  # class 소속 변수

    def __init__(self, memo):
        self.memo = memo
        self.creation_date = datetime.date.today()

        # global last_id
        Note.last_id += 1
        self.id = Note.last_id

    def match(self, txt):
        return txt in self.memo


def main():
    n1 = Note('hello first')
    n2 = Note('hello again')

    print(f'객체 n1 노트의 id : {n1.id}')
    print(f'객체 n2 노트의 id : {n2.id}')

    txt = 'hello'
    print(f'{txt}가 n1 노트에 존재하는 내용 여부 : {n1.match(txt)}')

    txt = 'second'
    print(f'{txt}가 n2 노트에 존재하는 내용 여부 : {n2.match(txt)}')


main()