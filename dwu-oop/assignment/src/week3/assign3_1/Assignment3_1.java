package week3.assign3_1;

import java.util.Scanner;

public class Assignment3_1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("책의 권수>>");
        int num = sc.nextInt();
        sc.nextLine(); // 개행문자(엔터)를 제거하기 위해 추가
        Book[] books = new Book[num];

        for (int i = 0; i < books.length; i++) {
            System.out.print("제목>>");
            String title = sc.nextLine();
            System.out.print("저자>>");
            String author = sc.nextLine();

            books[i] = new Book(title, author);
        }

        for (Book b: books) {
            System.out.println(b.toString());
        }

        System.out.println();
        System.out.print("찾으려는 책의 제목은 >> ");
        String search = sc.nextLine();
        boolean find = false;

        for (int i = 0; i < books.length; i++) {
            if (books[i].getTitle().equals(search)) {
                System.out.println("저자는 : " + books[i].getAuthor());
                find = true;
                break;
            }
        }

        if (!find) {
            System.out.println("찾으려는 책이 없습니다.");
        }

        sc.close();
    }
}
