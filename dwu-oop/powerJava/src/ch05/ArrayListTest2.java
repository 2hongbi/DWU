package ch05;

import java.util.ArrayList;

public class ArrayListTest2 {
    public static void main(String[] args) { // p. 168
        ArrayList<Person> list = new ArrayList<Person>();

        list.add(new Person("홍길동", "01012345678"));
        list.add(new Person("김유신", "01012345679"));
        list.add(new Person("최자영", "01012345680"));
        list.add(new Person("김영희", "01012345681"));

        for (Person obj : list) {
            System.out.println("(" + obj.name + "." + obj.tel + ")");
        }
    }
}
