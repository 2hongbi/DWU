package ch05.week3;

public class Week3_2 {
    public static void main(String[] args) {
        TV[] tvArr;
        tvArr = new TV[3];

        for (int i = 0; i < tvArr.length; i++) {
            tvArr[i] = new TV();
            tvArr[i].channel = 10;
            tvArr[i].channelUp();
            System.out.println("tvArr[" + i + "].channel = " + tvArr[i].channel);
        }
    }
}
