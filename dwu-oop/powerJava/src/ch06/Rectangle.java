package ch06;

public class Rectangle extends Shape{ // p.187
    int width, height;

    double calcArea() {
        return width * height;
    }

    void draw() {
        // 부모 클래스의 protected 멤버는 사용할 수 있음
        System.out.println("(" + x + "," + y + ") 위치에 가로 : " + width + " 세로 : " + height);
    }
}
