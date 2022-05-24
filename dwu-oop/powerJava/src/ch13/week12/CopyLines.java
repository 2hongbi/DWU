package ch13.week12;

import java.io.*;

public class CopyLines { // p.416
    public static void main(String[] args) {
        try (BufferedReader in = new BufferedReader(new FileReader("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/powerJava/src/ch13/week12/test.txt"));
             PrintWriter out = new PrintWriter(new FileWriter("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/powerJava/src/ch13/week12/output.txt"))) {
            String line;
            while ((line = in.readLine()) != null) {
                out.println(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
