package week8.assign8_2;
import java.util.InputMismatchException;
import java.util.Scanner;

public class Assignment8_2 {
    public static void main(String[] args) {
        char[] day = {'일', '월', '화', '수', '목', '금', '토'};

        while (true) {
            System.out.print("정수를 입력하세요>>");
            try {
                Scanner sc = new Scanner(System.in);
                int i = sc.nextInt();

                if (i == -2) {
                    System.out.println("프로그램 종료합니다...");
                    break;
                }

                int index = i % 7;
                System.out.println(day[index]);
            }catch (InputMismatchException e) {
                System.out.println("경고! 수를 입력하지 않았습니다.");
            }
        }
    }
}
