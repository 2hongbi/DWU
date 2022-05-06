package ch12.week10;

import java.util.HashSet;
import java.util.Scanner;

public class Week10_2 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        HashSet<String> hashSet = new HashSet<>();
        int i = 0;
        while (i < 5) {
            System.out.print("추가할 회원의 이름은 : ");
            String name = sc.next();
            hashSet.add(name);
            i++;
        }
        System.out.println(hashSet);
        sc.close();
    }
}
