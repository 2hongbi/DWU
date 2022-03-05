class Song:
    def __init__(self, title, artist):
        self.__title = title
        self.__artist = artist

    @property
    def title(self):
        return self.__title

    @property
    def artist(self):
        return self.__artist

    def __str__(self):
        return f'{self.__title} bv {self.__artist}'

    @classmethod
    def create_song(cls, slist):
        # 클래스 메소드 : 객체(인스턴스)의 소유가 아닌 클래스 소유 메쏘드 - 객체 (인스턴스)들이 공유
        # 특정 개체(인스턴스) 데이터 (속성)값을 변경할 수 없음
        songs = []

        for artist, title in slist:
            songs.append(cls(title, artist))    # cls = Song(class name)

        return songs


def main():
    songs = [('바다의 왕자', '레몬나인틴'), ('Ko Ko Pop', 'EXO')]

    # song_instance = []
    # for title, artist in songs:
    #    song_instance.append(Song(title, artist))
    song_instance = Song.create_song(songs)

    for song in song_instance:
        print(song)

main()