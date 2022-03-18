package ch05.week3;

public class TV { // Week3_2
    String color;
    boolean power;
    int channel;

    public void power() {
        power = !power;
    }

    public void channelUp() {
        channel ++;
    }

    public void channelDown() {
        channel --;
    }

}
