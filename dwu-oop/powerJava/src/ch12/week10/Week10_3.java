package ch12.week10;

import java.util.Scanner;
import java.util.TreeSet;

public class Week10_3 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        TreeSet<String> treeSet = new TreeSet<>();
        for (int i = 0; i < 5; i++) {
            System.out.print("추가할 회원의 이름은 : ");
            String name = sc.next();
            treeSet.add(name);
        }
        System.out.println(treeSet);
        sc.close();
    }
}
