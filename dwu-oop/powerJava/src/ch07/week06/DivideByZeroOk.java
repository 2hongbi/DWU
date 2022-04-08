package ch07.week06;

import java.util.Scanner;

public class DivideByZeroOk {
    public static void main(String[] args) {
        int x, y;
        Scanner sc = new Scanner(System.in);

        System.out.print("피젯수");
        x = sc.nextInt();
        System.out.print("젯수");
        y = sc.nextInt();

        try {
            int result = x / y; //예외 발생!
        } catch (ArithmeticException e) {

        }
    }
}
