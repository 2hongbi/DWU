package ch06;

abstract class ShapeA {
    int x, y;

    public void move(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public abstract void draw(); // 추상 메소드 선언. 추상 메소드를 하나라도 가지고 있으면 추상 클래스가 됨.
}
