package mid;

public class Hongbi extends Student {
    @Override
    void goToSchool() {
        System.out.println("아침에 학교 가기 싫다");
    }

    public static void main(String[] args) {
        Student student = new Hongbi();
    }
}
