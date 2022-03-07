package ch02;

import java.util.Scanner;

public class Exercise2 {
    public static void main(String[] args) {
        int v = 10;
        int k = v++%5;
        System.out.println(k);
        System.out.println(++v%5);

        int i1 = 10;
        int i2 = 20;
        String s1 = "9";
        System.out.println(i1 + i2 + s1);

        int x = 0;
        System.out.println(x);
        Scanner sc = new Scanner(System.in);
        x = sc.nextInt();
        System.out.println(x);
        x = sc.nextInt();
        System.out.println(x);
        boolean a = true, b = false, c = true;
        a = (b || c) && (a || false);
        System.out.println(a);
    }
}
