package ch13.week12;

import java.io.FileReader;
import java.io.IOException;

public class FileReaderExample2 {
    public static void main(String[] args) throws Exception {
        try (FileReader fr = new FileReader("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/powerJava/src/ch13/week12/test.txt")) {
            int ch;
            while ((ch = fr.read()) != -1) {
                System.out.print((char) ch);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
