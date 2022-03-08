package ch03;

import java.util.Scanner;

public class Grading { // p.70, 에제 3-2
    public static void main(String[] args) {
        int score;

        Scanner sc = new Scanner(System.in);
        System.out.print("성적을 입력하시오: ");
        score = sc.nextInt();

        if (score >= 95)
            System.out.println("학점 A+");
        else if (score >= 90)
            System.out.println("학점 A");
        else if (score >= 80)
            System.out.println("학점 B");
        else if (score >= 70)
            System.out.println("학점 C");
        else if (score >= 60)
            System.out.println("학점 D");
        else
            System.out.println("학점 F");
    }
}
