package ch07.week8;

public class StringCompare { // p.237
    public static void main(String[] args) {
        String s1 = "Apple";
        String s2 = "Apple";
        String s3 = new String("Apple");
        System.out.println(s1.equals(s2));
        System.out.println(s2.equals(s3));
    }
}
