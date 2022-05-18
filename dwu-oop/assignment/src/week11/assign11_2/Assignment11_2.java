package week11.assign11_2;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Assignment11_2 {
    public static String getKey(Map<String, Integer> map, int value) {
        for (Map.Entry<String, Integer> m : map.entrySet()) {
            if (m.getValue().equals(value)) {
                return m.getKey();
            }
        }
        return null;
    }

    public static void main(String[] args) {
        HashMap<String, Integer> nationMap = new HashMap<>();
        Scanner sc = new Scanner(System.in);
        for (int i = 0; i < 3; i++) {
            System.out.print((i+1) + ". 나라 이름, 인구 >> ");
            String nation = sc.next();
            int population = sc.nextInt();
            nationMap.put(nation, population);
        }

        Integer maxValue = Collections.max(nationMap.values());
        String nationValue = getKey(nationMap, maxValue);
        System.out.println("제일 인구가 많은 나라는 (" + nationValue + ", " + maxValue + ")");
    }
}
