package ch02;

import java.util.Scanner;

public class Exercise2_1 {
    // 하나의 상자에 오렌지를 10개씩 담을 수 있다고 하자. 오렌지가 127개가 있다면 상자 몇 개가 필요한가?
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("오렌지의 개수를 입력하시오: ");
        int orange_num = sc.nextInt();

        System.out.println((orange_num / 10) + "박스가 필요하고 " + (orange_num % 10) + "개가 남습니다.");
    }
}
