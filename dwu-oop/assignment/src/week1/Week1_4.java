package week1;

import java.util.Scanner;

public class Week1_4 {
    public static void main(String[] args) {
        String hexArray = "0123456789abcdef";

        Scanner sc = new Scanner(System.in);

        System.out.print("16진수 한 글자 입력 : ");
        String s = sc.next();

        s = s.toLowerCase();

        if (hexArray.contains(s)) {
            System.out.println("10진수 ===> " + hexArray.indexOf(s));
        } else {
            System.out.println("16진수가 아님");
        }

        sc.close();
    }
}
