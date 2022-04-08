package week4.assign4_1;

public class Student implements Comparable<Student>{
    String id;
    String name;

    public Student(String id, String name) {
        this.id = id;
        this.name = name;
    }

    @Override
    public String toString() {
        return "학번 = " + id + ", 이름 = " + name;
    }


    @Override
    public int compareTo(Student s) {
        return (this.id.compareTo(s.id)); // s의 id가 크면 양수, 같으면 0, 작으면 음수를 리턴
    }
}
