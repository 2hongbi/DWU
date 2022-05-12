package week10.assign10_2;

public class People {
    String name;
    int age;

    public People(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    public String toString() {
        return "이름 : " + name + ", 나이 : " + age;
    }
}
