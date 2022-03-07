package ch02;

public class BitOperator { // p.58, 예제 2-12
    public static void main(String[] args) {
        int x = 0x0fff;
        int y = 0xfff0;
        System.out.printf("%x ", (x & y));
        System.out.printf("%x ", (x | y));
        System.out.printf("%x ", (x ^ y));
        System.out.printf("%x ", (~x));
    }
}
