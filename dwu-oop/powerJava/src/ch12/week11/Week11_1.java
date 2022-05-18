package ch12.week11;

import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class Week11_1 {
    public static void main(String[] args) {
        Map<String, Integer> studentMap = new HashMap<>();
        studentMap.put("김길동", 85);
        studentMap.put("홍길동", 90);
        studentMap.put("최길동", 80);
        studentMap.put("홍길동", 95);

        System.out.println("총 Entry 수: " + studentMap.size());

        for (Map.Entry<String, Integer> s : studentMap.entrySet()) {
            String key = s.getKey();
            Integer value = s.getValue();
            System.out.println("    " + key + " : " + value);
        }

        Integer maxValue = Collections.max(studentMap.values());

    }
}
