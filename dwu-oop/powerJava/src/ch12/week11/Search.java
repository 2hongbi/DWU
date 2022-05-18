package ch12.week11;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Search {
    public static void main(String[] args) {
        int key = 50;
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            list.add(i);
        }
        int index = Collections.binarySearch(list, key); // 이진 탐색하여 list에서 key의 위치 반환
        System.out.println("탐색의 반환값 = " + index);
    }
}
