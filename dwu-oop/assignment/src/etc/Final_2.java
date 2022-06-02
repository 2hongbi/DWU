package etc;

import java.util.HashMap;
import java.util.Scanner;

public class Final_2 {
    public static void main(String[] args) {
        MemberHashMap memberHashMap = new MemberHashMap();

        Scanner sc = new Scanner(System.in);
        System.out.println("** 포인트 관리 프로그램입니다 **");
        while (true) {
            System.out.print("1. 입력    2. 삭제    3. 출력    4. 종료    선택한 번호는 >>");
            int choice = sc.nextInt();
            if (choice == 1) {
                System.out.print("이름과 포인터 입력 >> ");
                String name = sc.next();
                int point = sc.nextInt();
                memberHashMap.addMember(new Member(name, point));
            } else if (choice == 2) {
                System.out.print("삭제하려는 이름은 >>");
                String temp = sc.next();
                memberHashMap.removeMember(temp);
            } else if (choice == 3) {
                memberHashMap.showAllMember();
            } else {
                break;
            }
        }
        sc.close();
    }
}
