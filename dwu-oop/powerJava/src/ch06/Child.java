package ch06;

public class Child extends Parent{ // p.193, Parent
    public void print() {
        super.print(); // 메소드 오버라이드
        System.out.println("자식 클래스의 print() 메소드");
    }
}
