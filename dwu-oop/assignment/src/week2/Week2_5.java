package week2;

import java.util.Scanner;

public class Week2_5 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String str = sc.next();

        String temp = "";
        for (int i = str.length() - 1; i >= 0; i--) {
            temp += str.charAt(i);
        }

        System.out.println(temp);
        sc.close();
    }
}
