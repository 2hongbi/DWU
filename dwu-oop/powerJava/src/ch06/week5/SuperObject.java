package ch06.week5;

public class SuperObject {
    protected String name;

    public void paint() {
        System.out.println("0.Super");
        draw();
    }

    public void draw() {
        System.out.println("1.Super");
        System.out.println(name);
    }
}
