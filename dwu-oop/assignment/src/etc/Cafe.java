package etc;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Cafe {
    public static void main(String[] args) {
        Map<String, Integer> menu = new HashMap<>();
        menu.put("에스프레소", 2000);
        menu.put("아메리카노", 2500);
        menu.put("카푸치노", 3000);
        menu.put("카페라테", 3500);

        System.out.println("메뉴는 에스프레소, 아메리카노, 카푸치노, 카페라테가 있습니다.");
        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("주문 >> ");
            String choice = sc.next();

            if (choice.equals("그만")) {
                break;
            }

            try {
                int price = menu.get(choice);
                System.out.println(choice + "는 " + price + "원 입니다.");
            } catch (Exception e) {
                continue;
            }
        }
    }
}
