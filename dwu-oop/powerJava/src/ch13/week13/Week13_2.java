package ch13.week13;

import java.io.*;

public class Week13_2 {
    public static void main(String[] args) {
        try (BufferedReader in = new BufferedReader(new FileReader("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/powerJava/src/ch13/week13/in.txt"));
             PrintWriter out = new PrintWriter(new FileWriter("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/powerJava/src/ch13/week13/out.txt"))) {
            String line;
            while ((line = in.readLine()) != null) {
                out.print(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
