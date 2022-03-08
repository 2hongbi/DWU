package ch04;
class Circle2 {
    int radius; // 반지름
    String color; // 색상

    double calcArea() {
        return 3.14 * radius * radius;
    }
;}
public class CircleTest {
    public static void main(String[] args) {
        Circle2 obj; // 참조 변수 선언
        obj = new Circle2(); // 객체 생성
        obj.radius = 100;
        obj.color = "blue"; // 객체의 필드 접근
        double area = obj.calcArea();
        System.out.println("원의 면적=" + area);
    }
}
