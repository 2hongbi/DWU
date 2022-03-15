package ch04;

class Test {
    // 가변 길이 인수
    void sub(int... v) {
        System.out.println("인수의 개수 : " + v.length);
        for (int x : v) {
            System.out.print(x + " ");
        }
        System.out.println();
    }
}

public class VarArgsTest {
    public static void main(String[] args) {
        TestA c = new TestA();
        c.sub(1);
        c.sub(2, 3, 4, 5, 6);
        c.sub();
    }
}
