package ch05.week4;

public class Student implements Comparable{
    private String name;
    private double gpa;

    public Student(String n, double g) {
        name = n;
        gpa = g;
    }

    public String getName() {
        return name;
    }

    public double getGpa() {
        return gpa;
    }

    public int compareTo(Object obj) {
        Student other = (Student) obj;
        if (gpa < other.gpa) return -1;
        else if (gpa > other.gpa) return 1;
        else return 0;
    }
}
