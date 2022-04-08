package week5.assign5_3;

import java.util.ArrayList;
import java.util.Scanner;

public class Assignment5_3 {

    static int pay(Customer c, int price) {
        return (int) c.calcPrice(price);
    }

    public static void main(String[] args) {
        /*
        // 수정 전 과제 5_3
        ArrayList<Customer> customers = new ArrayList<>();

        Customer customer1 = new Customer("10010", "이순신");
        Customer customer2 = new Customer("10020", "신사임당");
        Customer customer3 = new GoldCustomer("10030", "홍길동");
        Customer customer4 = new GoldCustomer("10040", "이율곡");
        Customer customer5 = new VIPCustomer("10050", "김유신", 12345);

        customers.add(customer1);
        customers.add(customer2);
        customers.add(customer3);
        customers.add(customer4);
        customers.add(customer5);

        System.out.println("====== 고객 정보 출력 ======");
        for (Customer customer : customers) {
            System.out.println(customer.getCustomerInfo());
        }

        System.out.println("====== 할인율과 보너스 포인트 계산 ======");
        int price = 10000;
        for (Customer customer : customers) {
            System.out.println(customer.getCustomerName() + " 님이 " + pay(customer, price) + "원 지불하셨습니다.");
            System.out.println(customer.getCustomerName() + " 님의 현재 보너스 포인트는 " + customer.getBonusPoint() + "점입니다.");
        }
        */

        Customer customer = null;
        Scanner sc = new Scanner(System.in);
        System.out.println(">> 고객의 등급은? (s/v/g)");
        String grade = sc.nextLine();

        System.out.println(">> 고객의 이름은?");
        String name = sc.nextLine();

        if (grade.equals("s")) {
            customer = new Customer("10010", name);
        } else if (grade.equals("v")) {
            customer = new VIPCustomer("10010", name, 12345);
        } else if (grade.equals("g")) {
            customer = new GoldCustomer("10010", name);
        } else {
            System.out.println("정확한 등급을 입력해주세요.");
        }

        System.out.println(">> 물건의 가격은?");
        int price = sc.nextInt();

        customer.calcPrice(price);

        sc.close();
    }
}
