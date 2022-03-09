package ch03;

public class PizzaTopping { // p.95, 예제 3-14
    public static void main(String[] args) {
        String[] toppings = {"Pepperoni", "Mushrooms", "Onions", "Sausage", "Bacon"};

        for (int i = 0; i < toppings.length; i++) {
            System.out.print(toppings[i] + " ");
        }
    }
}