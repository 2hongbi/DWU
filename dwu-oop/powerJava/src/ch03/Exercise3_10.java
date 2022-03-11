package ch03;

public class Exercise3_10 {
    public static void main(String[] args) {
        double max = 0, sum = 0;
        double[] list = {1.0, 2.0, 3.0, 4.0};
        for (double ele: list) {
            sum += ele;
            if (max < ele) {
                max = ele;
            }
        }

        System.out.println("합은 " + sum);
        System.out.println("최대값은 " + max);
    }
}
