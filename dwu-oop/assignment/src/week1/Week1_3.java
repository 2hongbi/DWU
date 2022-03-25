package week1;

import java.util.Scanner;

public class Week1_3 {
    public static void main(String[] args) {
        int coin500, coin100, coin50, coin10;
        Scanner sc = new Scanner(System.in);

        System.out.print("500원 동전 개수 : ");
        coin500 = sc.nextInt();

        System.out.print("100원 동전 개수 : ");
        coin100 = sc.nextInt();

        System.out.print("50원 동전 개수 : ");
        coin50 = sc.nextInt();

        System.out.print("10원 동전 개수 : ");
        coin10 = sc.nextInt();

        int sum = 500 * coin500 + 100 * coin100 + 50 * coin50 + 10 * coin10;
        System.out.println("금액은 " + sum + "원 입니다.");

        sc.close();
    }
}
