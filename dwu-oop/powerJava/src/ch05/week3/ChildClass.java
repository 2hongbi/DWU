package ch05.week3;

public class ChildClass extends ParentClass{ // Week3_3
    int data = 200;

    public void print() {
        System.out.println("서브클래스 메소드");
        System.out.println(this.data);
        System.out.println(super.data);
    }

    public ChildClass() {
        System.out.println("child 생성자");
    }
}
