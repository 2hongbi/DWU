package week3.assign3_3;

import java.util.Scanner;

public class Assignment3_3 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        Book[] books = new Book[2];

        for (int i = 0; i < books.length; i++) {
            System.out.print("제목>>");
            String title = sc.nextLine();
            System.out.print("저자>>");
            String author = sc.nextLine();

            books[i] = new Book(title, author);
        }

        for (Book b:books) {
            System.out.println(b.toString());
        }

        sc.close();
    }
}
