package ch12.week9;

import java.util.Scanner;

public class Week9_4 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("5개의 정수를 입력하세요 >> ");

        Integer[] intList = new Integer[5];
        for (int i = 0; i < intList.length; i++) {
            intList[i] = sc.nextInt();
        }
        System.out.println("최대값: " + MyClass.getMax(intList));

        System.out.print("5개의 문자열을 입력하세요 >> ");
        String[] strList = new String[5];
        for (int i = 0; i < strList.length; i++) {
            strList[i] = sc.next();
        }
        System.out.println("최대값: " + MyClass.getMax(strList));
    }
}
