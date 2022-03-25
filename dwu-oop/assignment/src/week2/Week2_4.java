package week2;

class Average {
    public int getAverage(int a, int b) {
        return (a + b) / 2;
    }

    public int getAverage(int a, int b, int c) {
        return (a + b + c) / 3;
    }
}

public class Week2_4 {
    public static void main(String[] args) {
        Average avg = new Average();
        System.out.println("두수의 평균 : " + avg.getAverage(10, 20));
        System.out.println("세수의 평균 : " + avg.getAverage(20, 30, 40));
    }
}
