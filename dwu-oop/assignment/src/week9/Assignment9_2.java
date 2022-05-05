package week9;

import java.util.Scanner;
import java.util.StringTokenizer;

public class Assignment9_2 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.println("한줄로 문장을 입력하세요. (종료는 exit을 입력) >> ");
            String line = sc.nextLine();
            if (line.equals("exit")) {
                System.out.println("종료합니다...");
                break;
            }
            StringTokenizer st = new StringTokenizer(line, " ");
            int count = st.countTokens();
            System.out.println("어절 개수는 " + count);
            System.out.println("공백 개수는 " + (count - 1));
        }
        sc.close();
    }
}
