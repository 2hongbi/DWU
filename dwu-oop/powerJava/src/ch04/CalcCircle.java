package ch04;

public class CalcCircle { // p. 135
    int radius;

    public CalcCircle(int radius) {
        this.radius = radius; // this.radius는 필드이고, radius는 매개 변수라는 것을 알 수 있음
    }

    public CalcCircle() {
        this(0); // Circle(0) 호출
    }

    double calcArea() {
        return 3.14 * radius * radius;
    }
}
