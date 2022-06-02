package etc;

import java.util.HashSet;
import java.util.Scanner;

public class Final_1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        CustomerHashSet customerHashSet = new CustomerHashSet();
        while (true) {
            System.out.println("1. 회원 추가 | 2. 회원 삭제 | 3. 회원 검색 | 4. 회원 전체 출력 | 5. 종료");
            System.out.print("입력 >> ");
            int choice = sc.nextInt();
            sc.nextLine();
            if (choice == 1) {
                System.out.print("ID : ");
                String id = sc.nextLine();
                if (customerHashSet.checkEquals(id)) {
                    System.out.println("이미 존재하는 회원 아이디 입니다.");
                    continue;
                }
                System.out.print("NAME : ");
                String name = sc.nextLine();
                customerHashSet.addHashSet(new Customer(id, name));
            } else if (choice == 2 || choice == 3) {
                System.out.print("ID : ");
                String search = sc.next();
                customerHashSet.searchOrDelete(choice, search);
            } else if (choice == 4) {
                customerHashSet.showAll();
            } else {
                System.out.println("프로그램 종료");
                break;
            }
        }
        sc.close();
    }
}
