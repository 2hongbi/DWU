# 6. 역대 대통령의 대선 출마 선언문을 담은 speeches_presidents.csv를 이용하기

library(dplyr)
library(stringr)
library(readr)
library(tidytext)
library(KoNLP)

raw_speeches <- read_csv("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/speeches_presidents.csv")
raw_speeches

# 1. 4명의 대선 출마 선언문의 명사를 추출하여 로그 오즈비를 구하고, 
# 중요한 단어 10개씩 뽑아서 막대 그래프 그리기
speeches <- raw_speeches %>%
  mutate(value = str_replace_all(value, "[^가-힣]", " "),
         value = str_squish(value))

speeches

speeches <- speeches %>%
  unnest_tokens(input=value,
                output=word,
                token=extractNoun)
speeches

word_noun <- speeches %>%
  count(word, sort = T) %>%
  filter(str_count(word) > 1)

word_noun

top10 <- word_noun %>% head(10)

library(ggplot2)
ggplot(top10, aes(x = reorder(word, n), y = n)) +
  geom_col() +
  coord_flip() +
  labs(x = NULL)



frequency <- speeches %>%
  count(president, word) %>%
  filter(str_count(word) > 1)

frequency


frequency_wide <- frequency %>%
  pivot_wider(names_from = president, # 변수명으로 만들 값
              values_from = n,   # 변수에 채워 넣을 값
              values_fill = list(n=0))   # 결측치 0으로 변환
frequency_wide

# 로그 오즈비 구하기
frequency_wide <- frequency_wide %>%
  mutate(ratio_no = log((노무현 + 1) / (sum(노무현 + 1))),
         ratio_moon = log((문재인 + 1) / (sum(문재인 + 1))),
         ratio_park = log((박근혜 + 1) / (sum(박근혜 + 1))),
         ratio_lee = log((이명박 + 1) / (sum(이명박 + 1))))

# 2. speeches_presidents.csv를 불러와 이명박 전 대통령과 노무현 전 대통령의 연설문만을 추출하기
raw_speeches <- read_csv("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/speeches_presidents.csv")
raw_speeches

speeches <- raw_speeches %>%
  filter(president %in% c("이명박", "노무현")) %>%
  mutate(value = str_replace_all(value, "[^가-힣]", " "),
         value = str_squish(value))

speeches

# 2-1. 연설문에서 명사 추출해 다음 연설문 별 단어 빈도 구하기
speeches <- speeches %>%
  unnest_tokens(input = value,
                output = word,
                token = extractNoun)
speeches

frequency <- speeches %>%
  count(president, word) %>%
  filter(str_count(word) > 1)
frequency

# 2-2. 로그 RR를 이용해 두 연설문에서 상대적으로 중요한 단어를 10개씩 추출하기
frequency_wide <- frequency %>%
  pivot_wider(names_from = president,
              values_from = n,
              values_fill = list(n = 0))
frequency_wide

frequency_wide <- frequency_wide %>%
  mutate(log_odds_ratio = log(((이명박 + 1) / (sum(이명박 + 1))) /
                                ((노무현 + 1) / (sum(노무현 + 1)))))

frequency_wide

top10 <- frequency_wide %>%
  group_by(president = ifelse(log_odds_ratio > 0, "lee", "roh")) %>%
  slice_max(abs(log_odds_ratio), n = 10, with_ties = F)

top10

# 2-3. 두 연설문에서 상대적으로 중요한 단어를 나타낸 막대 그래프 그리기
ggplot(top10, aes(x = reorder(word, log_odds_ratio),
                  y = log_odds_ratio,
                  fill = president)) +
  geom_col() +
  coord_flip () +
  labs(x = NULL)