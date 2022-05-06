package week10.assign10_3;

public class Member implements Comparable{
    String name;

    public Member(String name) {
        this.name = name;
    }

    @Override
    public int compareTo(Object o) {
        Member other = (Member) o;
        return other.name.compareTo(this.name);
    }

    public String toString() {
        return name;
    }
}
