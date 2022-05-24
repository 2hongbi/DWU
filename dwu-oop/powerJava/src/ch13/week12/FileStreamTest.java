package ch13.week12;

import java.io.FileOutputStream;
import java.io.IOException;

public class FileStreamTest {    // p.411
    public static void main(String[] args) {
        byte list[] = {10, 20, 30, 40, 50, 60};
        try (FileOutputStream out = new FileOutputStream("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/powerJava/src/ch13/week12/test.bin")) {
            for (byte b : list) {
                out.write(b);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
