package ch13.week14;

public class ThreadX extends Thread{
    public void run() {
        for (int i = 0; i < 5; i++) {
            System.out.println(getName()); // Thread의 getName()
        }
    }
}
