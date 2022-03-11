package ch03;

public class Exercise3_6 { // p.109
    public static void main(String[] args) {
        System.out.print("2부터 100사이 모든 소수 : ");

        boolean check = false;
        for (int k = 2; k <= 100 ; k++) {
            for (int j = 2; j <= k-1; j++) {
                if (k % j == 0) {
                    check = false;
                    break;
                }
                check = true;
            }
            if (check) {
                System.out.print(k + " ");
                check = false;
            }
        }
    }
}
