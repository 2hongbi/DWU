package ch04;

public class Television { // p.133
    private int channel; // 채널 번호
    private int volume; // 볼륨
    private boolean onOff; // 전원 상태

    Television(int c, int v, boolean o) {
        channel = c;
        volume = v;
        onOff = o;
    }

    void print() {
        System.out.println("채널은 " + channel + "이고 볼륨은 " + volume + "입니다.");
    }

    public static void main(String[] args) {
        Television myTv = new Television(7, 10, true);
        myTv.print();

        Television yourTv = new Television(11, 20, true);
        yourTv.print();
    }
}
