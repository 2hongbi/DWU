package ch05;

public class Outer { // p.170
    private int num;

    class Inner {
        // 내부 클래스
        // 내부 클래스 안에서는 외부 클래스의 private 변수를 참조할 수 있음
        public void print() {
            System.out.println("여기는 내부 클래스입니다. ");
            System.out.println("num = " + num);
        }
    }

    void display() {
        // 내부 클래스도 사용하려면 객체를 생성해야 함
        Inner obj = new Inner();
        obj.print();
    }
}
