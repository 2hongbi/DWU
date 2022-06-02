package etc;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Lotto {
    public static void main(String[] args) {
        Map<Integer, String> lottoMap = new HashMap<>();
        Random random = new Random();

        while (lottoMap.size() < 11) {
            int num = random.nextInt(100) + 1;
            lottoMap.put(num, String.valueOf(num));
        }

        Object[] mapKey = lottoMap.keySet().toArray();
        Arrays.sort(mapKey);

        int[] result = new int[11];
        int i = 0;
        for (Object num : mapKey) {
            result[i] = (int) num;
            i++;
        }

        System.out.println("=== 경품 추첨 번호 : " + Arrays.toString(result));
    }
}
