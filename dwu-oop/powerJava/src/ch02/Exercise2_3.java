package ch02;

import java.util.Scanner;

public class Exercise2_3 {
    // 영수증에서는 10% 부가세와 잔돈 등이 인쇄되어 있다. 구입한 상품의 가격과 손님한테 받은 금액을 입력하면 부가세와 잔돈을 출력하는 프로그램을 작성하자.
    public static void main(String[] args) {
        int received_money, price;
        Scanner sc = new Scanner(System.in);

        System.out.print("받은 돈: ");
        received_money = sc.nextInt();

        System.out.print("상품 가격: ");
        price = sc.nextInt();

        double VAT = price * 0.1;
        System.out.println("부가세: " + (int) VAT);

        if (received_money > price) {
            System.out.println("잔돈: " + (received_money - price));
        }else {
            System.out.println("더 내야할 돈: " + (price - received_money));
        }
    }
}
