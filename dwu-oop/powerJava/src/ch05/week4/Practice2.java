package ch05.week4;

import java.util.Arrays;
import java.util.Scanner;

public class Practice2 {
    static int[] getValue() {
        Scanner sc = new Scanner(System.in);
        System.out.print("배열의 크기를 입력하세요 : ");
        int num = sc.nextInt();
        int[] values = new int[num];

        for (int i = 0; i < values.length; i++) {
            System.out.print("점수를 입력하세요 : ");
            values[i] = sc.nextInt();
        }
        sc.close();

        return values;
    }

    static int getAverage(int[] list) {
        int sum = 0;
        for (int i: list) {
            sum += i;
        }

        return sum / list.length;
    }

    public static void printArray(int[] list) {
        System.out.println(Arrays.toString(list));
    }

    public static void main(String[] args) {
        int[] values = getValue();
        printArray(values);
        System.out.println("점수 배열의 평균 : " + getAverage(values));
    }
}
