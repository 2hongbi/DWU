package week2;

import java.util.Scanner;

public class Week2_1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("정수를 입력하세요 : ");
        int[] num = new int[5];
        int sum = 0;
        for (int i = 0; i < num.length; i++) {
            num[i] = sc.nextInt();
        }
        for (int n:num) {
            sum += n;
        }
        System.out.println("합은 " + (double) sum);

        System.out.print("이름을 입력하세요 : ");
        String[] names = new String[5];
        for (int i = 0; i < names.length; i++) {
            names[i] = sc.next();
        }

        System.out.print("입력된 이름은 ");
        for (String name: names) {
            System.out.print(name + " ");
        }

        sc.close();
    }
}
