package ch13.week13;

import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;

public class Week13_1 {
    public static void main(String[] args) throws Exception{
        try (FileInputStream fileInputStream = new FileInputStream("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/powerJava/src/ch13/week13/week13.txt")) {
            int ch;
            while ((ch = fileInputStream.read()) != -1) {
                System.out.print((char) ch);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("\n\n");

        try (FileReader fileReader = new FileReader("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/powerJava/src/ch13/week13/week13.txt")) {
            int ch;
            while ((ch = fileReader.read()) != -1) {
                System.out.print((char) ch);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
