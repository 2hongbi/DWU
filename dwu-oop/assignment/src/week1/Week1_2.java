package week1;

import java.util.Scanner;

public class Week1_2 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String s;

        while (true) {
            System.out.print("문자를 입력하세요: ");
            s = sc.next();

            if (s.equals("0")) {
                break;
            }

            char c = s.charAt(0);
            if (Character.isUpperCase(c)) {
                System.out.println(Character.toLowerCase(c));
            } else if (Character.isLowerCase(c)) {
                System.out.println(Character.toUpperCase(c));
            } else if (Character.isDigit(c)) {
                System.out.println("영문자가 아닙니다.\n" + c);
            } else {
                System.out.println("올바른 문자가 아닙니다.\n" + c);
            }
        }

        sc.close();
    }
}
