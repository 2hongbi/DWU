package ch05;

import java.util.Scanner;

class Movie {
    String title, director;

    public Movie(String title, String director) {
        this.title = title;
        this.director = director;
    }
}

public class MovieArrayTest { // p.166
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        Movie[] list = new Movie[2];

        for (int i = 0; i < list.length; i++) {
            System.out.print("영화 제목 :");
            String title = sc.nextLine();

            System.out.print("영화 감독 : ");
            String director = sc.nextLine();

            list[i] = new Movie(title, director);
        }

        for (int i = 0; i < list.length; i++) {
            System.out.print("{ " + list[i].title + ", " + list[i].director + "} ");
        }
    }
}
