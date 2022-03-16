package ch04;

class Employee {
    private String name;
    private int salary;
    int age;

    // 생성자
    public Employee(String n, int a, int s) {
        name = n;
        age = a;
        salary = s;
    }

    // 직원의 이름을 반환
    public String getName() {
        return name;
    }

    // 직원의 월급을 반환
    private int getSalary() {
        return salary;
    }

    // 직원의 나이를 반환
    int getAge() {
        return age;
    }
}

public class EmployeeTest {
    public static void main(String[] args) {
        Employee e;
        e = new Employee("홍길동", 0, 3000);
        // e.salary = 300; // private variable -> error
        e.age = 26;
        // int sa = e.getSalary(); // private method -> error
        String s = e.getName();
        int a = e.getAge();
    }
}
