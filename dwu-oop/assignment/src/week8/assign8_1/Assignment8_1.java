package week8.assign8_1;

public class Assignment8_1 {
    public static void main(String[] args) {
        BankAccount ba = new BankAccount();

        try {
            ba.deposit(100);
            ba.withdraw(100);
            ba.withdraw(100);
        } catch (NegativeBalanceException e) {
            e.printStackTrace();
        }
    }
}
