package week10.assign10_2;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Scanner;

public class PeopleHashSet {
    private HashSet<People> peopleHashSet;

    public PeopleHashSet() {
        peopleHashSet = new HashSet<People>();
    }

    public void addHashSet(People people) {
        peopleHashSet.add(people);
    }

    public void showAll() {
        System.out.println("총 객체수 : " + peopleHashSet.size());
        for (People p: peopleHashSet) {
            System.out.println(p.toString());
        }
    }

    public void searchOrDelete(String check) {
        Scanner sc = new Scanner(System.in);
        Iterator<People> iterator = peopleHashSet.iterator();
        if (check.equals("search")) {
            System.out.print("찾는 이름을 입력 : ");
        } else if (check.equals("delete")) {
            System.out.print("삭제하려는 이름 입력 : ");
        }
        String name = sc.next();

        while (iterator.hasNext()) {
            People people = iterator.next();
            String n = people.getName();
            if (name.equals(n)) {
                if (check.equals("search")) {
                    System.out.println("찾는 데이터가 있다.");
                    break;
                } else if (check.equals("delete")) {
                    peopleHashSet.remove(people);
                    System.out.println(name + " 회원이 삭제됨");
                }
            }
        }
    }
}
