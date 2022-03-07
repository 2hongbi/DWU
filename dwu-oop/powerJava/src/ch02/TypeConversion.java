package ch02;

public class TypeConversion { // p.49, 예제 2-5
    public static void main(String[] args) {
        int i;
        double f;

        f = 5 / 4;
        System.out.println(f);

        f = (double) 5 / 4;
        System.out.println(f);

        i = (int) 1.3 + (int) 1.8;
        System.out.println(i);

    }
}
