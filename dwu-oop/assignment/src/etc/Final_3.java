package etc;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class Final_3 {
    public static void main(String[] args) {
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/assignment/src/etc/in.txt"));
             FileWriter fileWriter = new FileWriter("/Users/isoyeon/Documents/dwu/DWU/dwu-oop/assignment/src/etc/out.txt")) {
            String line = "";
            int num = 1;
            while ((line = bufferedReader.readLine()) != null) {
                fileWriter.write(" " + num + ": " + line + "\n");
                num++;
            }
            System.out.println("총 라인 수 : " + (num - 1));
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }
}
