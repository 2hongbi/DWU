package week3.assign3_2;

import java.util.Scanner;

public class Assignment3_2 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("잡지 이름은? ");
        String title = sc.nextLine();

        System.out.print("페이지수는? ");
        int pages = sc.nextInt();
        sc.nextLine();

        System.out.print("저자는? ");
        String author = sc.nextLine();

        System.out.print("발매일은? ");
        String pubDate = sc.nextLine();

        Magazine magazine = new Magazine(title, pages, author, pubDate);

        System.out.println(magazine.toString());

        sc.close();
    }
}
