package ch03;

import java.util.Scanner;

public class LoopExample2 { // p.78, 예제 3-6
    public static void main(String[] args) {
        int n;
        int i = 1;
        System.out.print("구구단 중에서 출력하고 싶은 단을 입력하시오: ");
        Scanner sc = new Scanner(System.in);
        n = sc.nextInt();

        while (i <= 9) {
            System.out.println(n + "*" + i + "=" + (n * i));
            i++;
        }
    }
}
