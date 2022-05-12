package week10.assign10_3;

import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;

public class Assignment10_3 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Set<Member> set = new TreeSet<>();
        System.out.println("3명의 이름을 입력 : ");
        for (int i = 0; i < 3; i++) {
            String name = sc.next();
            set.add(new Member(name));
        }
        System.out.println(set.toString());
        sc.close();
    }
}
