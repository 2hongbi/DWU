package ch02;

public class AreaTest { // p.45, 예제 2-2
    public static void main(String[] args) {
        final double PI = 3.141592;
        double radius, area;

        radius = 5.0;
        area = PI * radius * radius;
        System.out.println("반지름이 5인 원의 면적은 " + area);
    }
}