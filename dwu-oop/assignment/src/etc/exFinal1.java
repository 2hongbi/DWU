package etc;

import java.util.Iterator;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;

public class exFinal1 {
    public static void main(String[] args) {
        // TODO Auto-generated method stub
        Set<Members> s = new TreeSet<>();
        Scanner sc = new Scanner(System.in);
        int ch;
        String name, id, search, del;


        do {
            System.out.println("1.회원추가 | 2. 회원삭제 | 3. 회원검색 | 4. 회원 전체 출력 | 5. 종료");
            System.out.print("입력>> ");
            ch = sc.nextInt();
            if(ch == 1) {
                System.out.print("ID: ");
                id = sc.next();
                System.out.print("Name: ");
                name = sc.next();
                s.add(new Members(id, name));
                System.out.println(s.size());
            }
            else if(ch == 2) {
                int remove = 0;
                System.out.print("입력>> ");
                del = sc.next();
                Iterator<Members> it3 = s.iterator();

                while(it3.hasNext()) {
                    Members mDel = it3.next();

                    if(del.equals(mDel.id)) {
                        s.remove(mDel);
                        remove = 1;
                        System.out.println(del+" 회원이 삭제됨.");
                        break;
                    }
                }
                if (remove == 0)
                    System.out.println("존재하지 않는 아이디");
            }
            else if(ch == 3) {
                int flag = 0;
                int searchBoolean = 0;
                System.out.print("ID: ");
                search = sc.next();
                Iterator<Members> it = s.iterator();

                while(it.hasNext()) {
                    Members mSe = it.next();

                    if(search.equals(mSe.id)) {
                        searchBoolean = 1;
                        System.out.println("찾는 데이터가 있다.");
                        break;
                    }

                }

                if (searchBoolean == 0)
                    System.out.println("존재하지 않는 회원");
            }
            else if(ch == 4) {
                Iterator<Members> it1 = s.iterator();
                while(it1.hasNext()) {
                    Members ms = it1.next();
                    System.out.println(ms.toString());
                }

            }
            else
                System.out.println("잘못된 값");
        }while (ch != 5);
    }

}
