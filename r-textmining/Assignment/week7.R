# 문자열 벡터 읽기
origin_park<- readLines("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/speech_park.txt", encoding="UTF-8")
origin_park

library(dplyr)
library(stringr)

park <- origin_park %>%
  str_replace_all("[^가-힣]", " ") %>% # 한글만 
  str_squish() %>% # 연속된 공백 제거
  as_tibble()

park

# 토큰화하기
library(tidytext)
library(KoNLP)
word_noun <- park %>%
  unnest_tokens(input = value,
                output = word,
                token = extractNoun)

word_noun


# 자주 사용된 단어 20개 추출하기
top20 <- word_noun %>%
  count(word, sort = T) %>%
  filter(str_count(word) > 1) %>%
  head(20)

top20

# 그래프 그리기
library(ggplot2)
ggplot(top20, aes(x = reorder(word, n), y = n)) +
  geom_col() +
  coord_flip () +
  geom_text(aes(label = n), hjust = -0.3) +
  labs(x = NULL) + 
  theme(text=element_text(failmy="nanumgotihc"))
