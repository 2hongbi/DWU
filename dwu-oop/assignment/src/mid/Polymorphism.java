package mid;

public class Polymorphism {
    public static void f() {
        System.out.println("base class");
    }

    public static void main(String[] args) {
        Polymorphism po = new Derived();
        po.f();
    }

    static class Derived extends Polymorphism {
        public static void f() {
            System.out.println("sub class");
        }
    }
}
