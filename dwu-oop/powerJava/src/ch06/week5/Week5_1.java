package ch06.week5;

public class Week5_1 {
    static void checkPrice(Customer c) { // 매개변수의 다형성을 이용할 것!
        System.out.println(c.customerName+" 님의 할인율은 " + c.bonusRatio +"%이며, 적립금은 " + c.bonusPoint + "점 입니다.");
    }

    public static void main(String[] args) {
        Customer customer1 = new Customer();
        customer1.setCustomerID("10010");
        customer1.setCustomerName("이순신");
        System.out.println(customer1.getCustomerInfo());

        Customer customer2 = new VIPCustomer();
        customer2.setCustomerID("10020");
        customer2.setCustomerName("김유신");
        System.out.println(customer2.getCustomerInfo());
    }
}
