package week2;

public class Week2_3 {
    public static void main(String[] args) {
        System.out.println("random() 으로 발생한 수 : ");
        int sum = 0;

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                int num = (int) (Math.random() * 100 + 1);
                System.out.print(num + "  ");
                sum += num;
            }
            System.out.println();
        }

        System.out.println("합계 : " + sum);
        System.out.println("평균 : " + (double) (sum / 100));
    }
}
