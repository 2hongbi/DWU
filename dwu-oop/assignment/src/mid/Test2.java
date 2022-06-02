package mid;
class A {
    private void A() {
        System.out.println("A");
    }
}

class B extends A {
    public void A() {
        System.out.println("B");
    }
}

class C {
    public C() {
        System.out.println("C");
    }
}

public class Test2 {
    public static void main(String[] args) {
        A a = new A();
    }
}
