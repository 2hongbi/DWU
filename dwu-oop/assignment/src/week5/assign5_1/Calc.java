package week5.assign5_1;

abstract class Calc {
    int a, b;

    public void setValue(int a, int b) {
        this.a = a;
        this.b = b;
    }

    abstract int calculate();
}
