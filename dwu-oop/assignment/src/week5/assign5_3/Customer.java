package week5.assign5_3;

public class Customer {
    protected String customerID;
    protected String customerName;
    protected String customerGrade;
    protected int bonusPoint;
    protected double bonusRatio;

    public Customer() {
        // System.out.println("Customer() 생성자 호출");
        this.customerGrade = "SILVER";
        this.bonusRatio = 0.01;
    }

    public Customer(String customerID, String customerName) {
        this(); // initialize customer
        this.customerID = customerID;
        this.customerName = customerName;
    }

    public double calcPrice(int price) {
        this.bonusPoint += price * bonusRatio;
        System.out.println(customerName +" 님의 할인금액은 0 원, 지불금액은 " + price + "원입니다. \n" +
                "적립되는 보너스포인트는 " + bonusPoint + "원입니다.");
        return price;
    }

    public String getCustomerInfo() {
        return customerName +" 님의 등급은 " + customerGrade + "이며, 보너스 포인트는 " + bonusPoint + "입니다.";
    }

    public void setCustomerID(String customerID) {
        this.customerID = customerID;
    }

    public String getCustomerID() {
        return customerID;
    }

    public void setCustomerName(String customerName) {
        this.customerName = customerName;
    }

    public String getCustomerName() {
        return customerName;
    }

    public void setCustomerGrade(String customerGrade) {
        this.customerGrade = customerGrade;
    }

    public String getCustomerGrade() {
        return customerGrade;
    }

    public void setBonusPoint(int bonusPoint) {
        this.bonusPoint = bonusPoint;
    }

    public int getBonusPoint() {
        return bonusPoint;
    }

}
