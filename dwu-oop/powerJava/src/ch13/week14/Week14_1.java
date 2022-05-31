package ch13.week14;

public class Week14_1 {
    public static void main(String[] args) {
        ThreadX t1 = new ThreadX();
        Runnable r = new ThreadR();
        Thread t2 = new Thread(r);

        t1.start();
        t2.start();
    }
}
