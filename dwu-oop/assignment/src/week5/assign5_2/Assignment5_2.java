package week5.assign5_2;

import java.util.Scanner;

public class Assignment5_2 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int choice;
        int[] seats = new int[5];

        for (int i = 0; i < seats.length; i++) { // 예약 초기화
            seats[i] = 0;
        }

        while (true) {
            System.out.println("-------------------------------");
            System.out.println("1  2  3  4  5");
            System.out.println("-------------------------------");

            int full = 0;
            for (int seat : seats) {
                System.out.print(seat + "  ");

                if (seat == 1) {
                    full += 1;
                }
            }
            System.out.println();
            System.out.println("-------------------------------");

            if (full == 5) {
                System.out.println("만석입니다.");
                break;
            }
            System.out.print("예약하려는 좌석번호를 입력하세요 : ");
            choice = sc.nextInt();
            if (seats[choice - 1] == 0) {
                seats[choice - 1] = 1;
                System.out.println("예약되었습니다.");
            } else {
                System.out.println("이미 예약된 자리입니다.");
            }
        }

        sc.close();
    }
}
