package ch03;

import java.util.Scanner;

public class Exercise3_8 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String calc;
        int a, b;

        System.out.print("연산을 입력하세요: ");
        calc = sc.next();

        System.out.print("숫자 2개를 입력하세요: ");
        a = sc.nextInt();
        b = sc.nextInt();

        if (calc.equals("+")) {
            System.out.printf("%d+%d=%d", a, b, a+b);
        }else if(calc.equals("-")) {
            System.out.printf("%d-%d=%d", a, b, a-b);
        }else if(calc.equals("*")) {
            System.out.printf("%d*%d=%d", a, b, a*b);
        }else if(calc.equals("/")) {
            System.out.printf("%d/%d=%d", a, b, a/b);
        }

        sc.close();
    }
}
