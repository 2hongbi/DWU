package test.gongdong;

import java.io.*;
import java.util.HashMap;
import java.util.Iterator;
import java.util.StringTokenizer;

class Main {
    public static void main(String[] args) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;

        int PW = Integer.parseInt(br.readLine());
        int num = Integer.parseInt(br.readLine());

        HashMap<Integer, Integer> generations = new HashMap<>();
        for (int i = 0; i < num; i++) {
            st = new StringTokenizer(br.readLine());
            int hosu = Integer.parseInt(st.nextToken());
            int bibun = Integer.parseInt(st.nextToken());
            generations.put(hosu, bibun);
        }

        String s = br.readLine();
        if (s.endsWith("*")) {
            st = new StringTokenizer(s, "*");
            int length = st.countTokens();
            if (length == 1) {
                int temp = Integer.parseInt(st.nextToken());
                if (temp == PW) {
                    String t = Integer.toString(temp);
                    System.out.println(t.substring(t.length() - 4, t.length()) + " OPEN");
                }
            } else {
                int h = Integer.parseInt(st.nextToken());
                int b = Integer.parseInt(st.nextToken());
                if (generations.containsKey(h)) {
                    int value = generations.get(h);
                    if (b == value) {
                        String bi = Integer.toString(b);
                        System.out.println(bi.substring(bi.length() - 4, bi.length()) + " OPEN");
                    }
                }
            }
        } else if (s.endsWith("!")) { // 호출
            System.out.println("0000 SECURITY");
        } else if (s.endsWith("#")){ // 취소
            System.out.println("0000 #");
        }
    }
}