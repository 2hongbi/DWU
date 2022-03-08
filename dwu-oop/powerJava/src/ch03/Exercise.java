package ch03;

import java.util.Scanner;

public class Exercise { // p.91, Mini project
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int answer = (int) (Math.random() * 100);
        int count = 0;
        int guess = 0;

        do {
            System.out.print("정답을 추측하여 보시오: ");
            guess = sc.nextInt();

            if (answer < guess)
                System.out.println("HIGH");
            else if (answer > guess)
                System.out.println("LOW");

            count++;
        } while (answer != guess);

        System.out.println("축하합니다. 시도횟수="+count);
    }
}
