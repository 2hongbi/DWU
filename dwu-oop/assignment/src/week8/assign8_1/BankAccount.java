package week8.assign8_1;

public class BankAccount {
    int balance = 0;

    public void setBalance(int balance) {
        this.balance = balance;
    }

    public void deposit(int amount) throws NegativeBalanceException{
        if (balance < 0) {
            throw new NegativeBalanceException("음수 입금액");
        }
        this.balance += amount;
        System.out.println("정상 입금 처리=> 입금액: " + amount + ", 잔액: " + this.balance);
    }

    public void withdraw(int amount) throws NegativeBalanceException{
        if (amount > balance) {
            throw new NegativeBalanceException("잔고 부족");
        }else if(amount < 0) {
            throw new NegativeBalanceException("잘못된 금액");
        }
        this.balance -= amount;
        System.out.println("정상 출금 처리=> 인출액: " + amount + ", 잔액: " + this.balance);
    }
}
