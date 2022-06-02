package etc;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Scanner;

public class CustomerHashSet {
    private HashSet<Customer> customerHashSet;

    public CustomerHashSet() {
        customerHashSet = new HashSet<>();
    }

    public void addHashSet(Customer customer) {
        customerHashSet.add(customer);
    }

    public void showAll() {
        for (Customer c : customerHashSet) {
            System.out.println(c.toString());
        }
    }

    public boolean checkEquals(String id) {
        Iterator<Customer> iterator = customerHashSet.iterator();
        while (iterator.hasNext()) {
            Customer customer = iterator.next();
            if (id.equals(customer.getCustomerId())) {
                return true;
            }
        }
        return false;
    }

    public void searchOrDelete(int choice, String search) {
        Iterator<Customer> iterator = customerHashSet.iterator();
        boolean check = false;
        while (iterator.hasNext()) {
            Customer customer = iterator.next();
            String id = customer.getCustomerId();
            if (search.equals(id)) {
                check = true;
                if (choice == 2) {
                    customerHashSet.remove(customer);
                    System.out.println("삭제 완료");
                } else {
                    System.out.println("Name : " + customer.getName());
                }
                break;
            }
        }

        if (!check) {
            System.out.println("해당 회원은 존재하지 않습니다.");
        }
    }
}
