package ch03;

public class ArrayTest1 { // p.94
    public static void main(String[] args) {
        int[] s = new int[10];

        for (int i = 0; i < s.length; i++) {
            s[i] = i;
        }

        for (int i = 0; i < s.length; i++) {
            System.out.print(s[i] + " ");
        }
    }
}
