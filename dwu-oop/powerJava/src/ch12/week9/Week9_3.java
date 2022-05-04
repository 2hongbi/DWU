package ch12.week9;

import java.util.Scanner;

public class Week9_3 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Point<Integer> p = new Point<>();

        System.out.print("X를 입력하세요 >> ");
        int x = sc.nextInt();
        p.setX(x);

        System.out.print("Y를 입력하세요 >> ");
        int y = sc.nextInt();
        p.setY(y);

        System.out.println("X: " + p.getX());
        System.out.println("Y: " + p.getY());
    }
}
