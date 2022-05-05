package week9;

import java.util.Scanner;

public class Assignment9_5 {
    static <T> void a(T t) {
        System.out.println(t + "의 클래스 이름 : " + t.getClass());
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("정수를 입력하세요>> ");
        Integer integer = sc.nextInt();

        System.out.print("실수를 입력하세요>> ");
        Float flo = sc.nextFloat();

        a(integer);
        a(flo);
    }
}
