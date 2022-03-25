package ch05.week4;

import java.util.Arrays;
import java.util.Scanner;

public class MainTest2 {
    public static void main(String[] args) {
        int[] arr = {10, 7, 1, 23, 9};
        System.out.println("정렬 전 : " + Arrays.toString(arr));
        Arrays.sort(arr);
        System.out.println("정렬 후 : " + Arrays.toString(arr));

        String[] str = {"yes", "bcd", "no", "abc"};
        System.out.println("정렬 전 : " + Arrays.toString(str));
        Arrays.sort(str);
        System.out.println("정렬 후 : " + Arrays.toString(str));

        Scanner sc = new Scanner(System.in);
        String[] strList = new String[3];
        for (int i = 0; i < strList.length; i++) {
            System.out.print("문자열을 입력하세요>>");
            strList[i] = sc.nextLine();
        }
        Arrays.sort(strList);
        System.out.println(Arrays.toString(strList));
    }
}
