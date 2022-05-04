package ch12.week9;

public class MyClass <T> {
    T val;

    void set(T a) {
        val = a;
    }

    T get() {
        return val;
    }
}
