package ch05;

class Car{
    public int speed; // 속도
    public int gear; // 기어
    public String color; // 색상

    public Car() {
        speed = 0;
        gear = 1;
        color = "red";
    }

    public void speedUp() { // 속도 증가 메소드
        speed += 10;
    }

    public String toString() {
        return "속도: " + speed + " 기어: " + gear + "색상: " + color;
    }
}

public class CarArrayTest {
    public static void main(String[] args) {
        final int NUM_cARS = 5;
        Car[] cars = new Car[NUM_cARS];

    }
}
