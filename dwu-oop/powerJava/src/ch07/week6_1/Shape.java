package ch07.week6_1;

public class Shape implements Movable{
    protected int x, y;

    @Override
    public void move(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public void draw() {
        System.out.println("shape 그리기");
    }
}
