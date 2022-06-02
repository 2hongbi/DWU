package etc;

import week10.assign10_1.Student;

import java.util.Objects;

public class Members implements Comparable{
    String name;
    String id;

    Members(String id, String name) {
        this.name = name;
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public String toString() {
        return  "아이디: "+id+"이름 : "+name;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Members members = (Members) o;
        return Objects.equals(id, members.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public int compareTo(Object o) {
        Members other = (Members) o;
        return (this.id.compareTo(other.id));
    }


}
