package ch03;

import java.util.Scanner;

public class EvenOdd { // p.69, 에제 3-1
    public static void main(String[] args) {
        int number;

        Scanner sc = new Scanner(System.in);
        System.out.print("정수를 입력하시오: ");
        number = sc.nextInt();

        if (number % 2 == 0) {
            System.out.println("입력된 정수는 짝수입니다.");
        } else {
            System.out.println("입력된 정수는 홀수입니다.");
        }

        System.out.println("프로그램이 종료되었습니다.");
    }
}
