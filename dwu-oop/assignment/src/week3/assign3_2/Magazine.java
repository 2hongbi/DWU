package week3.assign3_2;

public class Magazine extends Book{
    String pubDate;

    public Magazine(String title, int pages, String author, String pubDate) {
        super(title, pages, author);
        this.pubDate = pubDate;
    }

    public void setPubDate(String pubDate) {
        this.pubDate = pubDate;
    }

    public String getPubDate() {
        return pubDate;
    }

    @Override
    public String toString() {
        return "책 이름 : " + title + '\n' +
                "페이지수 : " + pages + '\n' +
                "저자 : " + author + '\n' +
                "발매일 : " + pubDate;
    }
}
