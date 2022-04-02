package ch06.week5;

public class VIPCustomer extends Customer{
    private int agentID; // 고객 상담원 아이디
    double saleRatio;

    public VIPCustomer() {
        this.customerGrade = "VIP";
        this.bonusRatio = 0.05;
        this.saleRatio = 0.9;
        this.agentID = 12345;
    }

    public VIPCustomer(String customerID, String customerName) {
        this.customerID = customerID;
        this.customerName = customerName;
    }

    public double calcPrice(int price) {
        this.bonusPoint += price * bonusRatio;
        return price * saleRatio;
    }

    public int getAgentID() {
        return agentID;
    }
}
