package week3.assign3_3;

public class Book {
    String title, author;

    public Book(String title, String author) {
        this.title = title;
        this.author = author;
    }

    @Override
    public String toString() {
        return "(" + title + ", " + author + ")";
    }
}
