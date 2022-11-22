package test.treasure;

import java.io.*;
import java.util.Arrays;

class Main {
    public static int[] findKey(int[] array, char key) {
        for (int i = 0; i < array.length; i++) {
            if (array[i] == (int) key) {
                return new int[]{i, array.length - i};
            }
        }
        return new int[]{0, 0};
    }

    public static void main(String[] args) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String input = br.readLine();

        int[] dial = new int[input.length()];
        for (int i = 0; i < dial.length; i++) {
            dial[i] = (int) input.charAt(i);
        }

        String key = br.readLine();
        boolean toggle = false;
        int[] answer = {0, 0, 0}; // total, left, right
        int curr = 0;
        for (int i = 0; i < key.length(); i++) {
            int[] temp = findKey(dial, key.charAt(i)); // 좌우 이동
            int left = Math.abs(curr - temp[0]);
            int right = Math.abs(curr - temp[1]);
            curr = temp[0]; // curr update
            if (!toggle) {
                if (left < right) {
                    answer[0] += left + 1;
                    answer[1] += left;
                    toggle = true;
                } else {
                    answer[0] += right + 1;
                    answer[2] += right;
                }
            } else {
                if (left < right) {
                    answer[0] += left + 1;
                    answer[2] += left;
                    toggle = false;
                } else {
                    answer[0] += right + 1;
                    answer[1] += right;
                }
            }
        }

        for (int a : answer) {
            System.out.print(a + " ");
        }
    }
}