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

# mac 한글 꺠짐
theme_set(theme_gray(base_family = 'NanumGothic'))

# 그래프 그리기
library(ggplot2)
ggplot(top20, aes(x = reorder(word, n), y = n)) +
  geom_col() +
  coord_flip () +
  geom_text(aes(label = n), hjust = -0.3) +
  labs(x = NULL)


# 전처리하지 않은 연설문에서 연속된 공백을 제거하고 tibble 구조로 변환 후, 문장 기준으로 토큰화
sen_park <- origin_park %>%
  str_squish() %>% # 연속된 공백 제거
  as_tibble() %>% # tibble 변환
  unnest_tokens(input=value, output=sentence, token="sentences")

sen_park


# 연설문에서 "경제"가 사용된 문장 출력하기
sen_park %>% filter(str_detect(sentence, "경제"))
