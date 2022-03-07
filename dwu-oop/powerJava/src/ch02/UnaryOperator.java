package ch02;

public class UnaryOperator {
    public static void main(String[] args) {
        int x = 1;
        int y = 1;

        int nextx = ++x; // x의 값이 사용되기 전에 증가된다. nextx는 2가 된다.
        int nexty = y++; // y의 값이 사용된 후에 증가된다. nexty는 1이 된다.

        System.out.println(nextx);
        System.out.println(nexty);
    }
}
