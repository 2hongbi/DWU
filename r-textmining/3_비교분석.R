library(dplyr)

# 문재인 대통령 연설문 불러오기
raw_moon <- readLines("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/speech_moon.txt", encoding="UTF-8")
moon <- raw_moon %>%
  as_tibble() %>%
  mutate(president = "moon")

# 박근혜 대통령 연설문 불러오기
raw_park <- readLines("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/speech_park.txt", encoding="UTF-8")
park <- raw_park %>%
  as_tibble() %>%
  mutate(president = "park")

# 데이터 합치기
bind_speeches <- bind_rows(moon, park) %>%
  select(president, value)
head(bind_speeches)
tail(bind_speeches)

# 기본적인 전처리
library(stringr)
speeches <- bind_speeches %>%
  mutate(value=str_replace_all(value, "[^가-힣]", " "),
         value=str_squish(value))

speeches

# 토큰화
library(tidytext)
library(KoNLP)

speeches <- speeches %>%
  unnest_tokens(input = value,
                output = word,
                token = extractNoun)
speeches 

# 하위 집단별 단어 빈도 구하기 - count()
frequency <- speeches %>%
  count(president, word) %>% # 연설문 및 단어별 빈도
  filter(str_count(word) > 1)

head(frequency)

# 자주 사용된 단어 추출하기 - slice_max()
top10 <- frequency %>%
  group_by(president) %>% # president별로 분리
  slice_max(n, n = 10) # 상위 10개 추출

top10 %>%
  filter(president == "park") %>%
  print(n = Inf) # tibble의 모든 행 출력