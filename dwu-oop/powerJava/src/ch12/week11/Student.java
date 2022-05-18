package ch12.week11;

public class Student implements Comparable<Student>{
    int number;
    String name;

    public Student (int number, String name) {
        this.number =number;
        this.name = name;
    }

    public String toString() {
        return name;
    }

    public int compareTo(Student s) {
        return number - s.number;
    }
}
