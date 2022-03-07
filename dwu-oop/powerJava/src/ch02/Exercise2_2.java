package ch02;

import java.util.Scanner;

public class Exercise2_2 {
    // 마일을 킬로미터로 변환하는 프로그램을 작성하라. 1마일 = 1.609킬로미터
    public static void main(String[] args) {
        double kilometer;
        Scanner sc = new Scanner(System.in);
        System.out.print("마일을 입력하시오: ");
        kilometer = sc.nextDouble();

        double mile = kilometer * 1.609;
        System.out.println(kilometer + "마일은 " + mile + "킬로미터입니다.");
    }
}
