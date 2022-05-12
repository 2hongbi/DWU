package week10.assign10_1;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

public class Assignment10_1 {
    public static void main(String[] args) {
        ArrayList<Student> studentArrayList = new ArrayList<>();
        studentArrayList.add(new Student("홍길동", 1, 100, 100, 100));
        studentArrayList.add(new Student("송중기", 2, 90, 60, 80));
        studentArrayList.add(new Student("김자바", 3, 70, 80, 75));
        studentArrayList.add(new Student("이자바", 4, 80, 75, 90));
        studentArrayList.add(new Student("안자바", 5, 90, 60, 100));

        Collections.sort(studentArrayList);

        Iterator<Student> iterator = studentArrayList.iterator();
        while (iterator.hasNext()) {
            Student s = iterator.next();
            System.out.println(s.toString());
        }
    }
}
