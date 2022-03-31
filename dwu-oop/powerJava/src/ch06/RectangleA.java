package ch06;

public class RectangleA extends ShapeA{ // p.196, ShapeA
    int width, height;

    public void draw() { // 추상 메소드 구현
        System.out.println("사각형 그리기 메소드");
    }
}
