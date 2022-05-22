# 과제7: 역대 대통령의 취임사를 담은 inaugural_address.csv를 이용해 문제를 해결해 보세요.
raw_speeches <- read_csv("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/inaugural_address.csv")
raw_speeches

library(dplyr)
library(stringr)
library(tidytext)
library(KoNLP)

# 1. inaugural_address.csv를 불러와 분석에 적합하게 전처리하고 연설문에서 명사를 추출하세요.
speeches <- raw_speeches %>%
  mutate(value = str_replace_all(value, "[^가-힣]", " "),
         value = str_squish(value))

speech

speeches <- speeches %>%
  unnest_tokens(input = value,
                output = word,
                token = extractNoun)
speeches

# 2. TF-IDF를 이용해 각 연설문에서 상대적으로 중요한 단어를 10개씩 추출하세요.
frequency <- speeches %>%
  count(president, word) %>%
  filter(str_count(word) > 1)

frequency

frequency <- frequency %>%
  bind_tf_idf(term = word,           # 단어
              document = president,  # 텍스트 구분 변수
              n = n) %>%             # 단어 빈도
  arrange(-tf_idf)

frequency

top10 <- frequency %>%
  group_by(president) %>%
  slice_max(tf_idf, n = 10, with_ties = F)
top10

# 각 연설문에서 상대적으로 중요한 단어를 나타낸 막대 그래프를 만드세요.
library(ggplot2)
ggplot(top10, aes(x = reorder_within(word, tf_idf, president),
                  y = tf_idf,
                  fill = president)) +
  geom_col(show.legend = F) +
  coord_flip () +
  facet_wrap(~ president, scales = "free", ncol = 2) +
  scale_x_reordered() +
  labs(x = NULL)