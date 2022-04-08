package week4.assign4_3;

import java.util.Scanner;

public class Assignment4_3 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String[][] members = new String[0][2];

        while(true) {
            System.out.println("--------------------------------------------");
            System.out.println("1.회원수 | 2.정보입력 | 3. 회원리스트 | 4. 종료");
            System.out.println("--------------------------------------------");
            System.out.print("선택> ");

            int option = sc.nextInt();
            sc.nextLine();
            if (option == 1) {
                System.out.print("회원수> ");
                int size = sc.nextInt();
                sc.nextLine();
                members = new String[size][2];
            }else if (option == 2) {
                for (int i = 0; i < members.length; i++) {
                    System.out.print("이름을 입력하세요 : ");
                    members[i][0] = sc.nextLine();
                    System.out.print("전화번호를 입력하세요 : ");
                    members[i][1] = sc.nextLine();
                }
            }else if (option == 3) {
                System.out.println();
                System.out.println("** 회원 정보 **");
                for (String[] m:members) {
                    System.out.println(m[0] + " " + m[1]);
                }
            }else {
                break;
            }
        }
        sc.close();
    }
}
