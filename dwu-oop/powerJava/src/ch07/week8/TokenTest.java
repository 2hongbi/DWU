package ch07.week8;

public class TokenTest { // p.239
    public static void main(String[] args) {
        String[] tokens = "I am a boy.".split(" ");
        for (String token : tokens) {
            System.out.println(token);
        }

        String[] tokens2 = "100,200,300".split(",");
        for (String token : tokens2) {
            System.out.println(token);
        }
    }
}
