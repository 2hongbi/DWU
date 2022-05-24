package week13.assign13_1;

import java.io.*;
import java.util.Locale;

public class Assignment13_1 {
    public static void main(String[] args){
        try (BufferedReader in = new BufferedReader(new FileReader("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/assignment/src/week13/assign13_1/in.txt"));
             PrintWriter out = new PrintWriter(new FileWriter("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/assignment/src/week13/assign13_1/out.txt"))) {
            String line;
            while ((line = in.readLine()) != null) {
                out.println(line.toUpperCase());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
