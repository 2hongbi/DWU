package mid;

class Drawing implements Drawable {

    @Override
    public void draw(int x) {
        System.out.println();
    }
}

public class DrawableTest {

    void paint(Drawable d) {
        System.out.println("Paint!");
    }

    public static void main(String[] args) {
        Drawing drawing = new Drawing();
        DrawableTest test = new DrawableTest();
        test.paint(null);
        test.paint(drawing);
    }
}
