package ch06;

public class CircleTest { // p.184, Circle, Shape
    public static void main(String[] args) {
        Circle obj = new Circle(10);
        System.out.println("원의 중심 : (" + obj.x +"," + obj.y + ")");
        System.out.println("원의 면적 : " + obj.calcArea());
    }
}
