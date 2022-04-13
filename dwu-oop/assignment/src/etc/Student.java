package etc;

public abstract class Student {
    String name;
    String major;

    public Student() {
        this.name = "홍길동";
        this.major = "컴퓨터과학과";
    }

    abstract void goToSchool();
}