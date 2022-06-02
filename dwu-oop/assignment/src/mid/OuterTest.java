package mid;

class OuterClass {
    int value = 10;

    class InnerClass {
        int value = 20;

        void change() {
            int value = 30;
            System.out.println(value);
            System.out.println(this.value);
            System.out.println(OuterClass.this.value);
        }
    }
}
public class OuterTest {
    public static void main(String[] args) {
        OuterClass outer = new OuterClass();
        OuterClass.InnerClass innerClass = outer.new InnerClass();

        innerClass.change();
    }
}
