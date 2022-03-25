package week1;

import java.util.Scanner;

public class Week1_1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int sum = 0, count = 0;

        System.out.println("정수를 입력하고 마지막에 0을 입력하세요");
        while(true) {
            int num = sc.nextInt();
            if(num == 0) {
                break;
            }
            sum += num;
            count++;
        }
        System.out.println("입력한 수의 개수: " + count);
        System.out.println("평균: " + (double) (sum / count));

        sc.close();
    }
}
