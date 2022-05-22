# 과제8: "news_comment_BTS.csv"에는 2020년 9월 21일 방탄소년단이 '빌보드 핫 100 차트' 1위에 오른 소식을 다룬 기사에 달린 댓글이 들어있습니다. "news_comment_BTS.csv"를 이용해 다음의 문제를 해결해 보세요.

library(readr)
library(dplyr)
library(stringr)
library(textclean)
install.packages("textclean")

# Q1. "news_comment_BTS.csv"를 불러온 다음 행 번호를 나타낸 변수를 추가하고 분석에 적합하게 전처리하세요.
raw_comment <- read_csv("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/news_comment_BTS.csv")
glimpse(raw_comment)

news_comment <- raw_comment %>%
  mutate(id = row_number(),
         reply = str_squish(replace_html(reply)))

news_comment %>%
  select(id, reply)

# Q2. 댓글을 띄어쓰기 기준으로 토큰화하고 감성 사전을 이용해 댓글의 감정 점수를 구하세요.
library(tidytext)
library(KoNLP)

comment <- news_comment %>%
  unnest_tokens(input = reply,
                output = word,
                token = "words",  # 띄어쓰기 기준
                drop = F)  # 원문 유지
comment %>% select(word)

dic <- read_csv("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/knu_sentiment_lexicon.csv")

comment <- comment %>%
  left_join(dic, by = "word") %>%
  mutate(polarity = ifelse(is.na(polarity), 0, polarity))

comment %>%
  select(word, polarity) %>%
  arrange(-polarity)

score_comment <- comment %>%
  group_by(id, reply) %>%
  summarise(score = sum(polarity)) %>%
  ungroup()

score_comment %>%
  select(score, reply) %>%
  arrange(-score)

# Q3. 감정 범주 별 댓글 빈도를 나타낸 막대 그래프를 만드세요.
score_comment <- score_comment %>%
  mutate(sentiment = ifelse(score >=  1, "pos",
                            ifelse(score <= -1, "neg", "neu")))

score_comment %>%
  select(sentiment, reply)

frequency_score <- score_comment %>%
  count(sentiment)

frequency_score

library(ggplot2)
ggplot(frequency_score, aes(x = sentiment, y = n, fill = sentiment)) +
  geom_col() +
  geom_text(aes(label = n), vjust = -0.3)

# Q4. 댓글을 띄어쓰기 기준으로 토큰화한 다음 감정 범주 별 단어 빈도를 구하세요.
comment <- score_comment %>%
  unnest_tokens(input = reply,
                output = word,
                token = "words",
                drop = F)

frequency_word <- comment %>%
  count(sentiment, word, sort = T)

frequency_word

# Q5. 로그 RR을 이용해 긍정 댓글과 부정 댓글에 상대적으로 자주 사용된 단어를 10개씩 추출하세요.
library(tidyr)
comment_wide <- frequency_word %>%
  filter(sentiment != "neu") %>%
  pivot_wider(names_from = sentiment,
              values_from = n,
              values_fill = list(n = 0))

comment_wide

comment_wide <- comment_wide %>%
  mutate(log_odds_ratio = log(((pos + 1) / (sum(pos + 1))) /
                                ((neg + 1) / (sum(neg + 1)))))

comment_wide

top10 <- comment_wide %>%
  group_by(sentiment = ifelse(log_odds_ratio > 0, "pos", "neg")) %>%
  slice_max(abs(log_odds_ratio), n = 10)

top10

# Q6. 긍정 댓글과 부정 댓글에 상대적으로 자주 사용된 단어 각각 10개씩을 선택하여 긍정과 부정이 대비되도록 막대 그래프를 만드세요.
ggplot(top10, aes(x = reorder(word, log_odds_ratio),
                  y = log_odds_ratio,
                  fill = sentiment)) +
  geom_col() +
  coord_flip() +
  labs(x = NULL)

# Q7. 'Q3'에서 만든 데이터를 이용해 '긍정 댓글에 가장 자주 사용된 단어'를 언급한 댓글을 감정 점수가 높은 순으로 10개를 출력하세요.
score_comment %>%
  filter(str_detect(reply, "국내")) %>%
  arrange(score) %>%
  select(reply)