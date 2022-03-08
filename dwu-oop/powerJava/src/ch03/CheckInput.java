package ch03;

import java.util.Scanner;

public class CheckInput { // p.80, 예제 3-8
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int month;

        do {
            System.out.print("올바른 월을 입력하시오 [1-12]: ");
            month = sc.nextInt();
        } while (month < 1 || month > 12);

        System.out.println("사용자가 입력한 월은 " + month);
    }
}
