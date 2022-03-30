package ch05;

public class Circle { // 얘재 5-7, p.170
    class Point {
        int x, y;

        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    int radius;
    Point center;

    public Circle(int radius, int x, int y) {
        this.radius = radius;
        this.center = new Point(x, y);
    }

    double calcArea() {
        return 3.14 * radius * radius;
    }
}
