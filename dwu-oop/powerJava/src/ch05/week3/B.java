package ch05.week3;

public class B extends A{ // Week3_4
    public B() {
        System.out.println("생성자 B");
    }

    public B(int x) {
        // super(x);
        System.out.println("매개변수 생성자 B");
    }
}
