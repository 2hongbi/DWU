package ch03;

import java.util.Scanner;

public class Exercise3_4 { // p.108
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int height, weight;
        System.out.print("키를 입력하세요: ");
        height = sc.nextInt();

        System.out.print("몸무게를 입력하세요: ");
        weight = sc.nextInt();

        double standard = (height - 100) * 0.9;
        if (standard > weight) {
            System.out.println("저체중입니다.");
        } else if (standard == weight) {
            System.out.println("표준체중입니다.");
        } else {
            System.out.println("과체중입니다.");
        }

        sc.close();
    }
}