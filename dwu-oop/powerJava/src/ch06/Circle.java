package ch06;

public class Circle extends Shape { // p.184, Shape
    int radius;

    public Circle(int radius) {
        this.radius = radius;
        x = 0;
        y = 0;
    }

    double calcArea() {
        return 3.14 * radius * radius;
    }
}
