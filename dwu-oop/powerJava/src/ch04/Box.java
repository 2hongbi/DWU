package ch04;

public class Box { // p. 135
    int width, height, depth;

    public static void main(String[] args) {
        Box b = new Box(); // 컴파일러가 자동으로 기본 생성자를 추가
        System.out.println("상자의 크기 : (" + b.width + "," + b.height + "," + b.depth + ")");
    }
}
