package ch04;

class A {
    private int a; // 전용
    int b; // 디폴트
    public int c; // 공용
}

public class TestA { // p.138
    public static void main(String[] args) {
        A obj = new A();

        // obj.a = 10; // 전용 멤버는 다른 클래스에서는 접근이 안됨
        obj.b = 20; // 디폴트 멤버는 접근 가능
        obj.c = 30; // 공용 멤버는 접근 가능
    }
}
