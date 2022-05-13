package ch12.week11;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Sort {
    public static void main(String[] args) {
        String[] sample = {"i", "walk", "the", "line"};
        List<String> list = Arrays.asList(sample);  // 배열을 리스트로 변경
        Collections.sort(list);
        System.out.println(list);
    }
}
