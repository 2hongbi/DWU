package ch13.week12;

import java.io.FileInputStream;
import java.io.IOException;

public class FileStreamTest2 { // p.412
    public static void main(String[] args) {
        byte list[] = new byte[6];
        try (FileInputStream out = new FileInputStream("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/powerJava/src/ch13/week12/test.bin")) {
            out.read(list);
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (byte b : list) {
            System.out.print(b + " ");
        }
        System.out.println();
    }
}
