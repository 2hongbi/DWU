package ch03;

public class CardGame { // p.104, Mini project
    public static void main(String[] args) {
        String[] cards = {"Clubs", "Diamonds", "Hearts", "Spades"};
        String[] number = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"};

        for (int i = 0; i < 5; i++) {
            int one = (int) (Math.random() * 4);
            int two = (int) (Math.random() * 12);
            System.out.printf("%d, %d", one, two);
            System.out.printf("%s의 %s \n", cards[one], number[two]);
        }
    }
}
