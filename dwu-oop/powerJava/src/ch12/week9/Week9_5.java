package ch12.week9;

import java.util.Scanner;

public class Week9_5 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("5개의 정수를 입력하세요 >> ");
        Integer[] intList = new Integer[5];
        for (int i = 0; i < intList.length; i++) {
            intList[i] = sc.nextInt();
        }

        MyClass.displayArray(intList);
    }
}
