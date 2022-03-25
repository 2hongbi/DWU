package week2;

import java.util.Scanner;

class Phone {
    private String name;
    private String tel;

    public Phone(String name, String tel) {
        this.name = name;
        this.tel = tel;
    }

    public Phone() {
        this("이름 없음", "010-0000-0000");
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setTel(String tel) {
        this.tel = tel;
    }

    public String getName() {
        return name;
    }

    public String getTel() {
        return tel;
    }

    public void print() {
        System.out.println(getName() + "의 번호는 " + getTel());
    }
}


public class Week2_6 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String name, tel;

        System.out.print("이름, 전화번호를 입력하세요. >> ");
        name = sc.next();
        tel = sc.next();
        Phone phone = new Phone(name, tel);
        phone.print();

        System.out.print("이름, 전화번호를 입력하세요. >> ");
        name = sc.next();
        tel = sc.next();
        Phone phone2 = new Phone(name, tel);
        phone2.print();

        sc.close();
    }
}