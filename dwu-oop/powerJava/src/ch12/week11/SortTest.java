package ch12.week11;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class SortTest {
    public static void main(String[] args) {
        Student array[] = {
                new Student(20090001, "김철수"),
                new Student(20090002, "이철수"),
                new Student(20090003, "박철수"),
        };

        List<Student> list = Arrays.asList(array);
        Collections.sort(list); // 역순 : Collections.sort(list, Collections.reverseOrder())
        System.out.println(list);


    }
}
