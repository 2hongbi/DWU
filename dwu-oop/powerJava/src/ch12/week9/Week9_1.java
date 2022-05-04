package ch12.week9;

import java.util.Scanner;
import java.util.StringTokenizer;

public class Week9_1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("comma(,) 구분자를 이용하여 한줄로 입력하세요. >> ");
        String input = sc.nextLine();

        String[] inputList = input.split(",");
        System.out.println("** split()");
        for (int i = 0; i < inputList.length; i++) {
            System.out.print(inputList[i] + " ");
        }
        System.out.println();
        System.out.println("token 수: " + inputList.length);

        int i = 0;
        StringTokenizer st = new StringTokenizer(input, ",");
        while (st.hasMoreTokens()) {
            System.out.print(st.nextToken() + " ");
            i++;
        }
        System.out.println();
        System.out.println("token 수: " + i);
    }
}
