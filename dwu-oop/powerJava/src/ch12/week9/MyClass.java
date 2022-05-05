package ch12.week9;

import java.util.Arrays;

public class MyClass <T> {
    T val;

    void set(T a) {
        val = a;
    }

    T get() {
        return val;
    }

    public static <T extends Comparable> T getMax(T[] a) {
        T max = a[0];

        for (int i = 1; i < a.length; i++) {
            if (max.compareTo(a[i]) < 0) {
                max = a[i];
            }
        }
        return max;
    }

    static <T> void displayArray(T[] a) {
        System.out.println(Arrays.toString(a));
    }
}
