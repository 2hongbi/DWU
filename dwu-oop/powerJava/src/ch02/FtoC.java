package ch02;

import java.util.Scanner;

public class FtoC { // p.61, Mini project
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("화씨온도를 입력하시오: ");
        float f_temp = sc.nextFloat();
        float c_temp = (f_temp - 32) * 5 / 9;
        System.out.println("섭씨온도는 " + c_temp);
    }
}
