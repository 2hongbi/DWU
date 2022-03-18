package ch05.week3;

public class ParentClass { // Week3_3
    int data = 100;

    public void print() {
        System.out.println("슈퍼클래스 메소드");
    }

    public ParentClass() {
        System.out.println("Parent 생성자");
    }
}
