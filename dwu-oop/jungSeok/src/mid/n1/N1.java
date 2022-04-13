package mid.n1;

import java.util.Scanner;

public class N1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("두 정수와 연산자를 입력하시오.(ex. 80 4 /)");
        int a = sc.nextInt();
        int b = sc.nextInt();
        String operator = sc.next();
        Calc calc;

        switch (operator) {
            case "+":
                calc = new Add();
                System.out.println(calc.calculate(a, b));
                break;
            default:
                System.out.println("잘못된 연산자입니다.");
                break;
        }
    }
}
