package ch02;

import java.util.Scanner;

public class Exercise2_4 {
    // 사용자로부터 구의 반지름을 입력받아 부피를 계산하여 출력하는 프로그램을 작성하라. 모두 실수형으로.
    public static void main(String[] args) {
        double radius;
        Scanner sc = new Scanner(System.in);

        System.out.print("구의 반지름: ");
        radius = sc.nextDouble();
        double volume = (radius * radius * radius) * 4 / 3;
        System.out.println("구의 부피: " + volume);
    }
}
