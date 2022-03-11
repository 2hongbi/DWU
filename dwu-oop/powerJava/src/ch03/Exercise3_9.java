package ch03;

import java.util.Scanner;

public class Exercise3_9 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int first = 0;
        int second = 1;

        System.out.print("출력할 항의 개수: ");
        int num = sc.nextInt();

        for (int i = 0; i < num; i++) {
            if (i == 0) {
                System.out.print(first + " ");
            }else if(i == 1) {
                System.out.print(second + " ");
            }else{
                int temp = second;
                second = first + second;
                first = temp;
                System.out.print(second + " ");
            }
        }
        sc.close();
    }
}
