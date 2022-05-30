# 과제9: "news_comment_BTS.csv"에는 2020년 9월 21일 방탄소년단이 '빌보드 핫 100 차트' 1위에 오른 소식을 다룬 기사에 달린 댓글이 들어있습니다. "news_comment_BTS.csv"를 이용해 문제를 해결해 보세요.
#Q1. "news_comment_BTS.csv"를 불러온 다음 행 번호를 나타낸 변수를 추가하고 분석에 적합하게 전처리하세요.
library(readr)
library(dplyr)
raw_comment <- read_csv("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/news_comment_BTS.csv")
glimpse(raw_comment)


library(stringr)
library(textclean)
news_comment <- raw_comment %>%
  select(reply) %>%
  mutate(id = raw_number(),
         reply=str_replace_all(reply, "[^가-힣]", " "),
         reply=str_squish(reply))
news_comment %>%
  select(id, reply)

# Q2. 댓글에서 명사, 동사, 형용사를 추출하고 “/”으로 시작하는 모든 문자를 “다”로 바꾸시오.
library(tidytext)
library(KoNLP)
comment_pumsa <- news_comment %>%
  unnest_tokens(input=reply,
                output=word,
                token=SimplePos22,
                drop=F)
comment_pumsa

library(tidyr)
comment_pumsa <- comment_pumsa %>%
  separate_rows(word, sep="[+]")
comment_pumsa %>%
  select(word, reply)

comment <- comment_pumsa %>%
  separate_rows(word, sep="[+]") %>%
  filter(str_detect(word, "/n|/pv|/pa")) %>%
  mutate(word=ifelse(str_detect(word, "/pv|/pa"),
                     str_replace(word, "/.*s", "다"),
                     str_remove(word, "/.*s"))) %>%
  filter(str_count(word) >= 2) %>%
  arrange(id)

comment %>% select(word, reply)

# Q3. 다음 코드를 이용하여 유사어를 통일한 다음 한 댓글이 하나의 행이 되도록 단어를 결합하시오.
comment <- comment %>%
  mutate(word, case_when(str_detect(word, "축하") ~ "축하",
                         str_detect(word, "방탄") ~ "자랑",
                         str_detect(word, "대단") ~ "대단",
                         str_detect(word, "자랑") ~ "자랑",
                         T ~ word))

one_line_comment <- comment %>%
  group_by(id) %>%
  summarise(sentence=paste(word, collapse = " "))
one_line_comment


# Q4. 댓글을 바이그램으로 토큰화 한 다음 바이그램 단어쌍을 분리하시오.
bigram_comment <- one_line_comment %>%
  unnest_tokens(input=sentence,
                output=bigram,
                token="ngrams",
                n=2)
bigram_comment

bigram_separated <- bigram_comment %>%
  separate(bigram, c("word1", "word2"), sep=" ")
bigram_separated

# Q5. 단어쌍 빈도를 구한 다음 네트워크 그래프 데이터를 생성하시오.
bigram_pair <- bigram_separated %>%
  count(word1, word2, sort = T) %>%
  na.omit()

bigram_pair

# 난수를 고정한 다음 네트워크 그래프 데이터를 만드시오.
# 빈도가 3이상인 단어쌍만 사용하시오.
# 연결중심성과 커뮤니티를 나타낸 변수를 추가하시오.
library(tidygraph)
set.seed(1234)

bigram_graph <- bigram_pair %>%
  filter(n >= 3) %>%
  as_tbl_graph(directed=F) %>%
  mutate(centrality=centrality_degree(),
         group=as.factor(group_infomap()))
bigram_graph


# Q6. 바이그램을 이용하여 네트워크 그래프를 만드시오
# 난수를 고정한 다음 네트워크 그래프를 만드시오.
# 레이아웃을 “fr”로 설정하시오.
# 연결중심성에 따라 노드 크기를 정하고, 커뮤니티로 노드 색깔이 다르게 설정하시오.
# 노드의 범례를 삭제하시오.
# 텍스트가 노드 밖에 표시되게 설정하고, 텍스트의 크기를 5로 설정하시오.

library(ggraph)
theme_set(theme_gray(base_family = 'NanumGothic'))
set.seed(1234)
ggraph(bigram_graph, layout="fr") +
  geom_edge_link() +
  geom_node_point(aes(size=centrality,
                      color=group),
                  show.legend = F) +
   geom_node_text(aes(label=name),
                  repel=T,
                  size=5) +
  theme_graph()



# 그래프 꾸미기
library(showtext)
font_add_google(name = "Nanum Gothic", family = "NanumGothic")

set.seed(1234)
ggraph(bigram_graph, layout = "fr") +         # 레이아웃
  
  geom_edge_link(color = "gray50",            # 엣지 색깔
                 alpha = 0.5) +               # 엣지 명암
  
  geom_node_point(aes(size = centrality,      # 노드 크기
                      color = group),         # 노드 색깔
                  show.legend = F) +          # 범례 삭제
  scale_size(range = c(4, 8)) +               # 노드 크기 범위
  
  geom_node_text(aes(label = name),           # 텍스트 표시
                 repel = T,                   # 노드밖 표시
                 size = 5,                    # 텍스트 크기
                 family = "NanumGothic") +    # 폰트
  
  theme_graph()                               # 배경 삭제
