package week9;

public class MyMath <T> {
    public static <T extends Number> double getAverage(T[] a) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i].doubleValue();
        }
        return sum / a.length;
    }

}
