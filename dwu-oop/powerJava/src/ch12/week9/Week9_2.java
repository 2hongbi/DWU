package ch12.week9;

public class Week9_2 {
    public static void main(String[] args) {
        MyClass<String> s = new MyClass<>();
        s.set("hello");
        System.out.println(s.get());

        MyClass<Integer> n = new MyClass<>();
        n.set(5);
        System.out.println(n.get());
    }
}
