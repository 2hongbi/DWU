package ch07.week6_1;

public class Rectangle extends Shape{
    int height, width;

    public void setHeight(int height) {
        this.height = height;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    @Override
    public void draw() {
        System.out.println("사각형 그리기");
    }

    public void moveInfo() {
        System.out.println("x: ");
    }
}
