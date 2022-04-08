package week5.assign5_1;

import java.util.Scanner;

public class Assignment5_1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Calc calc;
        System.out.print("두 정수와 연산자를 입력하시오>>");
        int a = sc.nextInt();
        int b = sc.nextInt();
        String operator = sc.next();

        switch (operator) {
            case "+":
                calc = new Add();
                calc.setValue(a, b);
                System.out.println(calc.calculate());
                break;
            case "-":
                calc = new Sub();
                calc.setValue(a, b);
                System.out.println(calc.calculate());
                break;
            case "*":
                calc = new Mul();
                calc.setValue(a, b);
                System.out.println(calc.calculate());
                break;
            case "/":
                calc = new Div();
                calc.setValue(a, b);
                System.out.println(calc.calculate());
                break;
            default:
                System.out.println("올바른 연산자를 입력하세요.");
                break;
        }
        sc.close();
    }
}
