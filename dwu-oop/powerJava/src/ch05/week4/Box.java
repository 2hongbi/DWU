package ch05.week4;

public class Box implements Comparable{
    private double volume = 0;

    public Box(double v) {
        volume = v;
    }

    public int compareTo(Object otherObject) {
        Box other = (Box) otherObject;
        if (this.volume < other.volume) return -1;
        else if (this.volume > other.volume) return 1;
        else return 0;
    }
}
