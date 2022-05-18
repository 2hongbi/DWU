package week11.assign11_1;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Assignment11_1 {
    public static void main(String[] args) {
        Map<String, String> map = new HashMap<>();
        map.put("java", "자바");
        map.put("school", "학교");
        map.put("map", "지도");

        System.out.println("*** keySet() ***");
        for (String key : map.keySet()) {
            System.out.println("단어 : " + key + ", 의미 : " + map.get(key));
        }

        System.out.println("*** entrySet() ***");
        for (Map.Entry<String, String> entry : map.entrySet()) {
            System.out.println("(" + entry.getKey() + ", " + entry.getValue() + ")");
        }

        System.out.println();
        System.out.println();

        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("찾고 싶은 단어 : ");
            String search = sc.next();
            if (search.equals("quit")) {
                System.out.println("프로그램 종료");
                break;
            }

            if (map.containsKey(search)) {
                System.out.println("단어의 의미는 " + map.get(search));
            } else {
                System.out.println("찾는 단어가 없습니다.");
            }
        }

        sc.close();
    }
}
