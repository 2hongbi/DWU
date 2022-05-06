package ch12.week10;

import java.util.ArrayList;
import java.util.Scanner;

public class Week10_1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        ArrayList<String> arrayList = new ArrayList<>();
        int i = 0;
        while (i < 4) {
            System.out.print("이름을 입력하세요 >>");
            String name = sc.next();
            arrayList.add(name);
            i++;
        }

        for (String item : arrayList) {
            System.out.print(item + " ");
        }
        sc.close();
    }
}
