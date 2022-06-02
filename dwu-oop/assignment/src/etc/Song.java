package etc;

import week10.assign10_2.People;

import java.util.HashSet;
import java.util.Scanner;

public class Song {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        HashSet<Customer> hashSet = new HashSet<>();

        while (true) {
            System.out.println("1. 고객 데이터 입력 | 2. 고객 데이터 출력 | 3. 종료");
            int choice = sc.nextInt();

            if (choice == 1) {
                System.out.print("고객 번호 >> ");
                String id = sc.next();
                System.out.print("고객 이름 >> ");
                String name = sc.next();
                hashSet.add(new Customer(id, name));
            } else if (choice == 2) {
                for (Customer c : hashSet) {
                    System.out.println(c.toString());
                }
            } else {
                System.out.println("=== 종료합니다. ===");
                break;
            }
        }

        sc.close();
    }
}
