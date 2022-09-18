library(KoNLP)
useNIADic()
useSejongDic()
library(xlsx)
library(dplyr)
library(tidytext)
library(stringr)
library(tidyr)
library(textclean)
library(widyr)
library(ggplot2)
library(ggwordcloud)
library(wordcloud)
library(RColorBrewer)
library(tidygraph)
library(tm)
library(topicmodels)
library(ldatuning)

install.packages("ldatuning")

raw_data <- read.xlsx("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/final_data.xlsx", 1) %>%
  mutate(id = row_number())
raw_data

raw_data <- as_tibble(raw_data) 
raw_data

names(raw_data)
names(raw_data) = c("작성자", "작성일", "포스팅수", "탑승구간",
                    "클래스", "여행날짜", "작성제목", "댓글내용")
raw_data

# 댓글 내용 불러오기 + 기본적인 전처리
data_comment <- raw_data %>%
  mutate('댓글내용'=str_replace_all(댓글내용, "[^가-힣]", " "),
         '댓글내용'=str_squish(댓글내용)) %>%
  distinct(댓글내용, .keep_all = T) %>%   # 중복 댓글 제거
  filter(str_count(reply, boundary("word")) >= 3)
data_comment


# 명사 추출
comment <- data_comment %>%
  unnest_tokens(input = 댓글내용,
                output = word,
                token = extractNoun,
                drop = F) %>%
  filter(str_count(word) > 1) %>%
  group_by(id) %>%
  distinct(word, .keep_all = T) %>%
  ungroup() %>%
  select(id, word)
comment


# 빈도가 높은 단어 제거하기
count_word <- comment %>%
  add_count(word) %>%
  filter(n <= 200) %>%
  select(-n)

# 불용어, 유의어 확인하기
count_word %>%
  count(word, sort = T) %>%
  print(n=200)

# 불용어 목록 만들기
stopword <- c("정도", "그렇다", "번째", "만큼", "였다", "동안", "금새")

# 불용어, 유의어 처리하기
count_word <- count_word %>%
  filter(!word %in% stopword) %>%
  mutate(word = recode(word,
                       "승무원들이" = "승무원들",
                       "좋았습니" = "좋다",
                       "있습니" = "있다",
                       "직원들이" = "직원들",
                       "서울에서" = "서울",
                       "서울에" = "서울",
                       "승무원분" = "승무원들",
                       "인천에서" = "인천",
                       "추천합니" = "추천",
                       "합니" = "합니다",
                       "한국에서" = "한국",
                       "감사합니" = "감사",
                       "변경다른" = "변경",
                       "인천으로" = "인천",
                       "같습니" = "같다",
                       "비행이" = "비행",
                       "한국에" = "한국",
                       "한국으로" = "한국",
                       "높았습니" = "높다"))
count_word


# 문서별 단어 빈도 구하기 - 초반에 정의한 count_word 사용
count_word_doc <- count_word %>%
  count(id, word, sort = T)
count_word_doc

# DTM 만들기 - cast_dtm()
dtm_comment <- count_word_doc %>%
  cast_dtm(document = id, term = word, value = n)
dtm_comment


as.matrix(dtm_comment[1:10, 1:10])

# 최적의 토픽 수 정하기
models <- FindTopicsNumber(dtm = dtm_comment,
                           topics = 2:20,
                           return_models = T,
                           control = list(seed = 1234))
models %>%
  select(topics, Griffiths2004)

# 최적 토픽 수 정하기
FindTopicsNumber_plot(models)    # x-토픽 수, y-성능지표


# LDA 모델 만들기 - LDA()
lda_model <- LDA(dtm_comment,
                 k = 7,  # 토픽 수
                 method = "Gibbs",   # 샘플링 방법, 가장 많이 쓰이는 Gibbs 사용
                 control = list(seed = 1234))
lda_model


# 모델 내용 확인
glimpse(lda_model)


# beta 추출하기
term_topic <- tidy(lda_model, matrix = "beta")
term_topic   # 인척이 토픽 1에 등장할 확률은 0.0000442, 토픽 2에 등장할 확률은 0.0000406

term_order <- term_topic[order(term_topic$beta, decreasing = TRUE), ] %>%
  head(10)
term_order

ggplot(term_order,
       aes(x=reorder(term, beta),
           y=beta,
           fill = term)) +
  geom_col(show.legend = F) +
  coord_flip() +
  geom_text(aes(label = beta),  # 문서 빈도 표시
            hjust = -0.2) +   # 막대 밖에 표시
  geom_text(aes(label = term),   # 주요 단어 표시
            hjust = 1.03,    # 막대 안에 표시
            col = "white",    # 색깔
            fontface = "bold",
            family = "NanumGothicOTF") +    # 폰트
  scale_y_continuous(expand = c(0,0),    # y축 막대 간격 줄이기
                     limits = c(0, 0.1)) +    # y축 범위
  labs(x=NULL)


# gamma 추출하기
doc_topic <- tidy(lda_model, matrix = "gamma")
doc_topic

doc_order <- doc_topic[order(doc_topic$gamma, decreasing = TRUE), ] %>%
  head(5)
doc_order

doc_order_join <- raw_data %>%
  right_join(doc_order, by = c("id" = "topic"))

doc_order_join

# gamma 그래프 그리기
ggplot(doc_order_join,
       aes(x=reorder(document, gamma),
           y=gamma,
           fill = gamma)) +
  geom_col(show.legend = F) +
  coord_flip() +
  geom_text(aes(label = document),  # 문서 빈도 표시
            hjust = -0.2) +   # 막대 밖에 표시
  geom_text(aes(label = gamma),   # 주요 단어 표시
            hjust = 1.03,    # 막대 안에 표시
            col = "white",    # 색깔
            fontface = "bold",
            family = "NanumGothicOTF") +    # 폰트
  scale_y_continuous(expand = c(0,0),    # y축 막대 간격 줄이기
                     limits = c(0, 0.4)) +    # y축 범위
  labs(x=NULL)

# 토픽별 단어 수
term_topic %>%
  count(topic)

# 문서별로 확률이 가장 높은 토픽 추출
doc_class <- doc_topic %>%
  group_by(document) %>%
  slice_max(gamma, n=1)
doc_class

# integer 변환
doc_class$document <- as.integer(doc_class$document)


            
# 결합 확인
data_comment_topic %>%
  select(id, topic)

# 토픽별 주요 단어 목록 만들기
top_terms <- term_topic %>%
  group_by(topic) %>%
  slice_max(beta, n=6, with_ties = F) %>%    # 확률이 동점인 단어 제외
  summarise(term = paste(term, collapse = ", "))   # 주요 단어를 한 행으로 합하기
top_terms

# 토픽별 문서 빈도 구하기
count_topic <- data_comment_topic %>%
  count(topic)
count_topic <- count_topic[1:7, ]


# 문서 빈도에 주요 단어 결합하기
count_topic_word <- count_topic %>%
  left_join(top_terms, by = "topic") %>%
  mutate(topic_name = paste("Topic", topic))
count_topic_word

# mac 한글 꺠짐
theme_set(theme_gray(base_family = 'NanumGothicOTF'))

install.packages("extrafont")
library(extrafont)

font_import()
fonts()

# 토픽별 문서 수와 주요 단어 
ggplot(count_topic_word,
       aes(x=reorder(topic_name, n),
           y=n,
           fill = topic_name)) +
  geom_col(show.legend = F) +
  coord_flip() +
  geom_text(aes(label = n),  # 문서 빈도 표시
            hjust = -0.2) +   # 막대 밖에 표시
  geom_text(aes(label = term),   # 주요 단어 표시
            hjust = 1.03,    # 막대 안에 표시
            col = "white",    # 색깔
            fontface = "bold",
            family = "NanumGothicOTF") +    # 폰트
  scale_y_continuous(expand = c(0,0),    # y축 막대 간격 줄이기
                     limits = c(0, 300)) +    # y축 범위
  labs(x=NULL)




# 원문에 토픽 번호 부여
data_comment_topic <- raw_data %>%
  left_join(doc_class, by = c("id" = "document"))
data_comment_topic


# 토픽별 주요 문서 추출
cmt_topic <- data_comment_topic %>%
  group_by(topic) %>%
  slice_max(gamma, n = 100)
cmt_topic


# 토픽 1 내용 살펴보기
cmt_topic %>%
  filter(topic == 6) %>%
  pull(댓글.작성내용.)

# 토픽 이름 목록 만들기
name_topic <- tibble(topic = 1:7,
                     name = c("1. 대한항공의 전반적인 서비스",
                              "2.", 
                              "3.",
                              "4.",
                              "5.",
                              "6. ",
                              "7. 클래스 불문하고 최상의 서비스를 제공하는 대한항공"))
