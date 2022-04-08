package week5.assign5_3;

public class GoldCustomer extends Customer{
    double saleRatio;

    public GoldCustomer(String customerID, String customerName) {
        super(customerID, customerName);
        customerGrade = "GOLD";
        bonusRatio = 0.02;
        saleRatio = 0.1;
    }

    public double calcPrice(int price) {
        bonusPoint += price * bonusRatio;
        int sale = price - (int) (price * saleRatio);
        System.out.println(customerName +" 님의 할인금액은 "+ (int) (price * saleRatio) +"원, 지불금액은 " + sale + "원입니다. \n" +
                "적립되는 보너스포인트는 " + bonusPoint + "원입니다.");
        return sale;
    }
}
