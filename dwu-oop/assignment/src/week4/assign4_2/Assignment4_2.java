package week4.assign4_2;

import java.util.Arrays;

public class Assignment4_2 {

    public static Person getMax(Person[] people) {
        Arrays.sort(people); // 내림차순 정렬
        return people[0];
    }

    public static void main(String[] args) {
        Person[] people = new Person[3];
        people[0] = new Person("홍길동", 180);
        people[1] = new Person("이길동", 170);
        people[2] = new Person("김길동", 190);

        System.out.println("[제일 키 큰 사람]");
        System.out.println(getMax(people));
    }
}
