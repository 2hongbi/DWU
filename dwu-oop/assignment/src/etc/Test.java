package etc;

class OOP extends OOP1 {

    int A = super.number;
    String B = super.name;
}

class OOP1 extends OOP2{
    int number=1;
}

class OOP2{
    String name="A";
}

class Test {
    public static void main(String[] args) {
        OOP oop  = new OOP();
        System.out.println(oop.A);
        System.out.println(oop.B);
    }
}
