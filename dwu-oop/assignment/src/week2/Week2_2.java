package week2;

import java.util.Scanner;

public class Week2_2 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String jumin, gyesok;

        while (true) {
            System.out.print("당신의 주민번호를 입력하세요. (012345-1234567) >> ");
            jumin = sc.next();

            char sex = jumin.charAt(7);
            if (sex == '1' || sex == '3') {
                System.out.println("당신은 남자입니다.");
            }else if (sex == '2' || sex == '4') {
                System.out.println("당신은 여자입니다.");
            }else {
                System.out.println("유효하지 않은 주민번호입니다.");
            }

            System.out.print("입력을 계속하시겠습니까? (y/n) ");
            gyesok = sc.next();

            if (gyesok.equals("n")) {
                break;
            }
        }

        sc.close();
    }
}
