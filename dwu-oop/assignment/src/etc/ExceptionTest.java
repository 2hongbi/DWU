package etc;

import java.io.IOException;

public class ExceptionTest {

    void test(int x) throws Exception {
        if (x < 0) {
            throw new Exception("0 이상의 수를 입력하세요!!!");
        }
        System.out.println(x / 2);
    }

    public static void main(String[] args) {
        try {
            new ExceptionTest().test(-1);
        }catch (Exception e) {
            System.out.println("예외 메시지 : " + e.getMessage());
            System.out.println("~~에러 발생~~");
        }finally {
            System.out.println("예외가 발생해도 프로그램은 계속 된다 ㅋ");
        }
    }
}
