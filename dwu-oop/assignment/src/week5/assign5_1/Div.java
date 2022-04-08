package week5.assign5_1;

public class Div extends Calc{
    int calculate() {
        if (a == 0 || b == 0) {
            System.out.println("0(으로/을) 나눌 수 없습니다.");
            return -1;
        }
        return a / b;
    }
}
