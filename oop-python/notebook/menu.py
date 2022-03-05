import sys
from notebook.notebooks import Notebook


class Menu:
    def __init__(self):
        self.notebook = Notebook()
        self.choices = {'1': self.show_notes,
                        '2': self.search_notes,
                        '3': self.add_notes,
                        '4': self.modify_note,
                        '5': self.quit}

    def __display_menu(self):
        print("""
*** Notebook Menu ***
1. Show all notes
2. Search Notes
3. Add notes
4. Modify note
5. Quit
        """)

    def run(self):
        while True:
            self.__display_menu()

            choice = input('Enter your choice :')
            action = self.choices.get(choice)

            if action:
                action()
            else:
                print(f'{choice} is not a valid choice.')

    def show_notes(self, notes=None):
        if notes is None:
            notes = self.notebook.notes
        elif not len(notes):
            print('No matched note!')

        for note in notes:
            print(f'{note.id} : {note.memo}')

    def search_notes(self):
        target = input('Search for : ')
        notes = self.notebook.search(target)
        self.show_notes(notes)

    def add_notes(self):
        memo = input('Enter a memo : ')
        self.notebook.new_note(memo)
        print('Your note has been added')

    def modify_note(self):
        id = int(input('Enter note id : '))
        memo = input('Enter memo: ')

        if memo:
            self.notebook.modify_memo(id, memo)

    def quit(self):
        print('Thank you for using notebook')
        sys.exit()


if __name__ == '__main__':
    menu = Menu()
    menu.run()