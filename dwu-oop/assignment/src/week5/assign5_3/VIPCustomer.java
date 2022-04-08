package week5.assign5_3;

public class VIPCustomer extends Customer{
    private int agentID; // 고객 상담원 아이디
    double saleRatio;

    public VIPCustomer(String customerID, String customerName, int agentID) {
        super(customerID, customerName);
        this.agentID = agentID;
        customerGrade = "VIP";
        bonusRatio = 0.05;
        saleRatio = 0.1;
    }

    public double calcPrice(int price) {
        bonusPoint += price * bonusRatio;
        int sale = price - (int) (price * saleRatio);
        System.out.println(customerName +" 님의 할인금액은 "+ (int) (price * saleRatio) +"원, 지불금액은 " + sale + "원입니다. \n" +
                "적립되는 보너스포인트는 " + bonusPoint + "원입니다.");
        return sale;
    }

    public int getAgentID() {
        return agentID;
    }

    public String getCustomerInfo() { // 고객 정보 출력 메서드 재정의
        return super.getCustomerInfo() + " 담당 상담원 번호는 " + agentID + "입니다.";
    }
}
