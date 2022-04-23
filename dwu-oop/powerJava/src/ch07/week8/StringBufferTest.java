package ch07.week8;

public class StringBufferTest { // p.240
    public static void main(String[] args) {
        String s1 = "Hello";
        String s2 = s1.concat(" World"); // s1의 내용을 변경하는 것이 아닌, 새로운 String 객체를 생성해서 반환함

        StringBuffer sb = new StringBuffer("Happiness depends upon ourselves");
        sb.append("Hello");
        System.out.println(sb);
    }
}
