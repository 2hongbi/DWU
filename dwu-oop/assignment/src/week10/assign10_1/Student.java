package week10.assign10_1;

public class Student implements Comparable{
    String name;
    int num;
    int korean;
    int english;
    int math;

    public Student(String name, int num, int korean, int english, int math) {
        this.name = name;
        this.num = num;
        this.korean = korean;
        this.english = english;
        this.math = math;
    }

    public int getTotal() {
        return (korean + english + math);
    }

    public double getAverage() {
        return (float)(korean + english + math) / 3;
    }

    public String toString() {
        return name + ", " + num + ": " + korean + ", " + english + ", " + math + ", " + getTotal() + ", " + getAverage();
    }


    @Override
    public int compareTo(Object o) {
        Student other = (Student) o;
        return (this.name.compareTo(other.name));
    }
}
