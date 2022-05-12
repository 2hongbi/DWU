package week10.assign10_2;

import java.util.HashSet;

public class Assignment10_2 {
    public static void main(String[] args) {
        HashSet<People> peopleHashSet = new HashSet<>();
        peopleHashSet.add(new People("이길동", 30));
        peopleHashSet.add(new People("이길동", 30));
        peopleHashSet.add(new People("홍길동", 30));
        System.out.println(peopleHashSet.toString());
    }
}
