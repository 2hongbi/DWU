package ch12.week9;

public class GenericTest <T> {
    private T a;

    public GenericTest(T a) {
        this.a = a;
    }

    public T getT() {
        return a;
    }

    public static void main(String[] args) {
        GenericTest<String> ts = new GenericTest<String>("제네릭 타입으로 String");
        System.out.println(ts.getT());

        GenericTest<Integer> ti = new GenericTest<>(100); // 자바 SE 7 버전 이후는 제네릭 클래스의 생성자 호출 시, 타입 인수를 구체적으로 주지 않아도 컴파일러가 문맥에서 타입을 추측함
        System.out.println(ti.getT());

        GenericTest<Double> td = new GenericTest<>(3.14);
        System.out.println(td.getT());

        GenericTest<?> tw1 = ts; // <?>는 모든 타입의 파라미터를 받을 수 있음
        System.out.println(tw1.getT());

        GenericTest<?> tw2 = ti;
        System.out.println(tw2.getT());

        GenericTest<?> tw3 = td;
        System.out.println(tw3.getT());
    }
}
