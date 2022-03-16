package ch05;

class Pizza {
    private String toppings;
    static int count = 0; // 정적 변수 선언

    public Pizza(String toppings) {
        this.toppings = toppings;
        count++;
    }
}

public class PizzaTest { // p.155
    public static void main(String[] args) {
        Pizza p1 = new Pizza("Super Supreme");
        Pizza p2 = new Pizza("Cheese");
        Pizza p3 = new Pizza("Pepperoni");
        int n = Pizza.count;
        System.out.println("지금까지 판메된 피자 개수 = " + n);
    }
}
