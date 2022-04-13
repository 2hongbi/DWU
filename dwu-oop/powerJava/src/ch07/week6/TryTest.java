package ch07.week6;

import java.io.FileReader;
import java.io.IOException;

public class TryTest {
    public static void main(String[] args) {
        try (FileReader fr = new FileReader("./test.txt")) {
            char[] a = new char[50];
            fr.read(a);
            for (char c : a) {
                System.out.print(c);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
