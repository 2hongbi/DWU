package ch05;

import java.util.Scanner;

public class MiniProject5 { // Mini project, p.172
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        User user;

        while(true) {
            System.out.println("==============================");
            System.out.println("1. Sign Up \n" +
                    "2. Login \n" +
                    "3. Print All Users \n" +
                    "4. Exit \n");
            System.out.println("==============================");
            System.out.println("번호를 입력하시오 : ");
            int select = sc.nextInt();

            if (select == 1) {
                System.out.println("id: ");
                String id = sc.next();
                System.out.println("Password : ");
                String pw = sc.next();
                user = new User(id, pw);
            }else if (select == 3) {
                continue;
            }else if (select == 4) {
                break;
            }
        }

    }
}
