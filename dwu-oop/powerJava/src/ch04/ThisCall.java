package ch04;

class People {
    String name;
    int age;

    People() {
        this("이름 없음", 31);
    }

    People(String name, int age) {
        this.name = name;
        this.age = age;
    }
}

public class ThisCall {
    public static void main(String[] args) {
        People noName = new People();
        System.out.println("이름: " + noName.name);
        System.out.println("나이: " + noName.age);
    }
}
