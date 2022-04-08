package week4.assign4_2;

public class Person implements Comparable{
    String name;
    int height;

    public Person(String name, int height) {
        this.name = name;
        this.height = height;
    }

    @Override
    public String toString() {
        return "이름 : " + name + ", height : " + (double) height;
    }

    @Override
    public int compareTo(Object o) {
        Person other = (Person) o;
        return (other.height - this.height);
    }
}
