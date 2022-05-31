package ch13.week14;

public class ThreadR implements Runnable{
    public void run() {
        for (int i = 0; i < 5; i++) {
            // 현재 실행중인 Thread 반환
            System.out.println(Thread.currentThread().getName());
        }
    }
}
