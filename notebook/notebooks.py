from notebook.note import Note


class Notebook:
    def __init__(self):
        self.notes = []

    def new_note(self, memo):
        note = Note(memo)
        self.notes.append(note)

    def __find_note(self, note_id):
        for note in self.notes:
            if note.id == note_id:
                return note
        return None

    def modify_memo(self, note_id, memo):
        target_note = self.__find_note(note_id)

        if target_note:
            target_note.memo = memo
            return True
        else:
            return False

    def search(self, txt):
        found_notes = [note for note in self.notes if note.match(txt)]
        return found_notes
