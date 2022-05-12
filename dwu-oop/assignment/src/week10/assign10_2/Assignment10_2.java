package week10.assign10_2;

public class Assignment10_2 {
    public static void main(String[] args) {
        PeopleHashSet peopleHashSet = new PeopleHashSet();
        peopleHashSet.addHashSet(new People("이길동", 30));
        peopleHashSet.addHashSet(new People("홍길동", 30));
        peopleHashSet.addHashSet(new People("홍길동", 30));

        peopleHashSet.showAll();

        peopleHashSet.searchOrDelete("search");
        peopleHashSet.searchOrDelete("delete");
    }
}
