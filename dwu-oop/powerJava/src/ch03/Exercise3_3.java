package ch03;

public class Exercise3_3 {
    public static void main(String[] args) {
        // first method
        for (int i = 1; i < 6; i++) {
            System.out.printf("(%d, %d), ", i, 6-i);
        }

        System.out.println();
        // second method
        int[] dice1 = {1, 2, 3, 4, 5, 6};
        int[] dice2 = {1, 2, 3, 4, 5, 6};
        for (int i = 0; i < dice1.length; i++) {
            for (int j = 0; j < dice2.length; j++) {
                if (dice1[i] + dice2[j] == 6) {
                    System.out.printf("(%d, %d), ", dice1[i], dice2[j]);
                }
            }
        }
    }
}
