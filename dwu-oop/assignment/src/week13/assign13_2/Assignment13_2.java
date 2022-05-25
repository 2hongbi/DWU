package week13.assign13_2;

import java.io.*;
import java.util.Scanner;

public class Assignment13_2 {
    public static void main(String[] args) throws IOException {
        Scanner sc = new Scanner(System.in);

        PrintWriter pw = new PrintWriter("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/assignment/src/week13/assign13_2/result.txt");
        while (true) {
            System.out.print("사용자 번호를 입력하세요 >> ");
            int num = sc.nextInt();
            System.out.print("사용자 이름을 입력하세요 >> ");
            String name = sc.next();
            System.out.print("사용자 전화번호를 입력하세요 >> ");
            String phoneNum = sc.next();
            System.out.print("사용자 이메일을 입력하세요 >> ");
            String email = sc.next();
            pw.write(num + "," + name + "," + phoneNum + "," + email);

            System.out.print("입력이 끝났으면 0, 계속 입력하려면 1을 입력하세요 >> ");
            int check = sc.nextInt();
            if (check == 0) {
                pw.flush();
                pw.close();
                break;
            } else if (check == 1) {
                pw.write("\r\n");
                continue;
            }
        }

        System.out.print("검색할 사용자 번호를 입력하세요 >> ");
        String search = sc.next();
        Scanner fileSc = new Scanner(new File("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/assignment/src/week13/assign13_2/result.txt"));

        boolean check = false;
        while (fileSc.hasNextLine()) {
            String line = fileSc.nextLine();
            Scanner sc2 = new Scanner(line).useDelimiter(",");

            while (sc2.hasNext()) {
               if (sc2.next().equals(search)) {
                   String[] strings = sc2.nextLine().split(",");
                   System.out.println("사용자 번호 " + search + "의 전화번호는 " + strings[2] + "입니다.");
                   check = true;
                   break;
               }
            }
            sc2.close();
        }

        if (!check) {
            System.out.println("해당 번호는 없는 사용자 번호입니다.");
        }

        sc.close();
        fileSc.close();
    }
}
