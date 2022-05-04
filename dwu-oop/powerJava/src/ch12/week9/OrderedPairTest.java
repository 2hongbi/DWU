package ch12.week9;

public class OrderedPairTest {
    public static void main(String[] args) {
        Pair<String, Integer> p1 = new OrderedPair<>("Even", 0);
        Pair<String, String> p2 = new OrderedPair<>("Hello", "world");

        System.out.println(p1.getKey() + ", " + p1.getValue());
        System.out.println(p2.getKey() + ", " + p2.getValue());
    }
}
