package week4.assign4_1;

import java.util.Arrays;
import java.util.Scanner;

public class Assignment4_1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("배얼의 크기는>>");
        int num = sc.nextInt();
        sc.nextLine();

        Student[] students = new Student[num];
        for (int i = 0; i < students.length; i++) {
            System.out.print("학번>>");
            String id = sc.nextLine();
            System.out.print("이름>>");
            String name = sc.nextLine();

            students[i] = new Student(id, name);
        }

        Arrays.sort(students);
        for (Student s:students) {
            System.out.println(s.toString());
        }
    }
}
