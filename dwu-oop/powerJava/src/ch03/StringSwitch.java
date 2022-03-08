package ch03;

import java.util.Scanner;

public class StringSwitch { // p.75, 예제 3-5
    public static void main(String[] args) {
        String month;

        Scanner sc = new Scanner(System.in);
        System.out.print("달의 이름을 입력하시오: ");
        month = sc.next();

        int monthNumber;
        switch (month) {
            case "january":
                monthNumber = 1;
                break;
            case "february":
                monthNumber = 2;
                break;
            case "march":
                monthNumber = 3;
                break;
            default:
                monthNumber = 0;
                break;
        }
        System.out.println(monthNumber);
    }
}
