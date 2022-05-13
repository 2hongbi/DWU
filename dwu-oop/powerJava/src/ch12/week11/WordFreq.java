package ch12.week11;

import java.util.HashMap;
import java.util.Map;

public class WordFreq {
    public static void main(String[] args) {
        Map<String, Integer> m = new HashMap<>();

        String[] sample = {"to", "be", "or", "not", "to", "be", "is", "a", "problem"};

        for (String a : sample) {
            Integer freq = m.get(a);
            m.put(a, (freq == null) ? 1 : freq + 1);
        }

        System.out.println(m.size() + "단어가 있습니다.");
        System.out.println(m.containsKey("to"));
        System.out.println(m.isEmpty());
        System.out.println(m);

    }
}
