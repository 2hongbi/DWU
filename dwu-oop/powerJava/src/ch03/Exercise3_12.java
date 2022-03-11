package ch03;

import java.util.Scanner;

public class Exercise3_12 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int sum = 0;

        for (int i = 0; i < 5; i++) {
            System.out.print("성적을 입력하세요: ");
            int score = sc.nextInt();
            sum += score;
        }
        System.out.println("합계 : " + sum);
        System.out.println("평균 : " + (double) (sum / 5));

        sc.close();
    }
}
