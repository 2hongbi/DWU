package ch12.week9;

public class Point <T> {
    T x;
    T y;

    void setX(T x) {
        this.x = x;
    }

    T getX() {
        return x;
    }

    void setY(T y) {
        this.y = y;
    }

    T getY() {
        return y;
    }
}
