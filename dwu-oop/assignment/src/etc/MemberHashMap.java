package etc;

import java.util.HashMap;
import java.util.Iterator;

public class MemberHashMap {
    private HashMap<String, Member> hashMap;

    public MemberHashMap() {
        hashMap = new HashMap<>();
    }

    public void addMember(Member member) {
        if (hashMap.containsKey(member.getName())) {
            Member m = hashMap.get(member.getName());
            m.addPoint(member.point);
        } else {
            hashMap.put(member.getName(), member);
        }
    }

    public boolean removeMember(String name) {
        if (hashMap.containsKey(name)) {
            hashMap.remove(name);
            System.out.println(name + "은(는) 삭제되었습니다.");
            return true;
        }
        System.out.println(name + "은(는) 등록되지 않은 사람입니다.");
        return false;
    }

    public void showAllMember() {
        Iterator<String> iterator = hashMap.keySet().iterator();
        while (iterator.hasNext()) {
            String key = iterator.next();
            Member member = hashMap.get(key);
            System.out.print(member.toString());
        }
        System.out.println();
    }
}
