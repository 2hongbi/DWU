package week9;

import java.util.*;

public class Assignment9_1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("단어 문자열을 입력하세요 >> ");
        String s = sc.nextLine();

        StringTokenizer st = new StringTokenizer(s, " ");
        int count = st.countTokens();
        String[] strings = new String[count];
        int i = 0;
        while (st.hasMoreTokens()) {
             strings[i] = st.nextToken();
             i++;
        }
        System.out.println("모두 "+ strings.length +"개의 단어가 있습니다.");
        System.out.println(" [분기된 토큰]");
        for (String str : strings) {
            System.out.println(str);
        }

        System.out.println(" 토큰수:" + strings.length);

        System.out.println("====== sort =======");
        Arrays.sort(strings);
        for (String str: strings) {
            System.out.println(str);
        }

        sc.close();
    }
}
