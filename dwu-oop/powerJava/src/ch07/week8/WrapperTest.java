package ch07.week8;

public class WrapperTest { // p. 232 ~ 234
    public static void main(String[] args) {
        Integer obj1 = new Integer(100);
        int value1 = obj1.intValue();

        Double obj2 = new Double(3.141592);
        double value2 = obj2.doubleValue();

        String s1 = Integer.toString(10);
        String s2 = Integer.toString(10000);
        String s3 = Float.toString(3.14f);
        String s4 = Double.toString(3.141592);

        System.out.println(value1);
        System.out.println(value2);
        System.out.println(s1);
        System.out.println(s2);
        System.out.println(s3);
        System.out.println(s4);

        int i = Integer.parseInt("10");
        long l = Long.parseLong("10000");
        float f = Float.parseFloat("3.14");
        double d = Double.parseDouble("3.141592");

        System.out.println(i);
        System.out.println(l);
        System.out.println(f);
        System.out.println(d);

        Integer box;
        box = 10;   // 정수를 자동으로 Integer 객체로 포장한다. (boxing)
        System.out.println(box + 1); // box는 자동으로 int형으로 변환(unboxing)
    }
}
