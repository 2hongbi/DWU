package week11.assign11_3;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Assignment11_3 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Map<String, String> phoneMap = new HashMap<>();

        System.out.print("입력하려는 데이터의 수 : ");
        int num = sc.nextInt();

        System.out.println("이름과 전화번호를 입력하세요. (예: 홍길동 010-111-1212)");
        for (int i = 0; i < num; i++) {
            System.out.print((i + 1) + ". 이름, 전화번호 >> ");
            String name = sc.next();
            String phoneNum = sc.next();
            phoneMap.put(name, phoneNum);
        }

        System.out.print("전화번호를 찾으려는 회원의 이름은 : ");
        String search = sc.next();
        boolean check = false;
        for (Map.Entry<String, String> m : phoneMap.entrySet()) {
            if (m.getKey().equals(search)) {
                System.out.println(search + "의 전화번호는 : " + phoneMap.get(search));
                check = true;
                break;
            }
        }

        if (!check) {
            System.out.println("해당하는 이름의 회원은 리스트에 없습니다.");
        }
        sc.close();
    }
}
