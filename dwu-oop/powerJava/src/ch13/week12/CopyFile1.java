package ch13.week12;

import java.io.FileReader;
import java.io.FileWriter;

public class CopyFile1 {
    public static void main(String[] args) throws Exception{
        try (FileReader fr = new FileReader("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/powerJava/src/ch13/week12/test.txt");
             FileWriter fw = new FileWriter("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/powerJava/src/ch13/week12/copy.txt")) {
            int c;
            while((c = fr.read()) != -1) {
                fw.write(c);
            }
        }
    }
}
