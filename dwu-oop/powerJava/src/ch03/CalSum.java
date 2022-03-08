package ch03;

public class CalSum { // p.79, 예제 3-7
    public static void main(String[] args) {
        int i = 1;
        int sum = 0;
        while (i <= 10) {
            sum += i;
            i++;
        }
        System.out.println("합계="+sum);
    }
}
