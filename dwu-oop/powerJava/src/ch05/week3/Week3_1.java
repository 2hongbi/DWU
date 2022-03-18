package ch05.week3;

import java.util.Arrays;

public class Week3_1 {
    public static void main(String[] args) {
        int[] iArr = {10, 20, 30, 40, 50};
        String[] iStr = {"a", "b", "c", "d", "e"};

        System.out.println("** for-each문으로 출력");
        for (int i: iArr) {
            System.out.println(i);
        }

        for (String s: iStr) {
            System.out.println(s);
        }

        System.out.println("** Arrays.toString으로 출력");
        System.out.println("iArr : " + Arrays.toString(iArr));
        System.out.println("iStr : " + Arrays.toString(iStr));

        System.out.println("** println으로 출력");
        System.out.println(iArr);
        System.out.println(iStr);
    }
}
