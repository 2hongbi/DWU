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

# 빈도 동점 단어 제외하고 추출하기 - slice_max(with_ties = F)
top10 <- frequency %>%
  group_by(president) %>%
  slice_max(n, n = 10, with_ties = F)

top10

# mac 한글 꺠짐
theme_set(theme_gray(base_family = 'NanumGothic'))

# 막대 그래프 만들기
library(ggplot2)

# 변수의 항목별로 그래프 만들기 - facet_wrap()
ggplot(top10, aes(x = reorder(word, n),
                  y = n,
                  fill = president)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~president)

# 그래프별로 y축 설정하기
ggplot(top10, aes(x = reorder(word, n),
                  y = n,
                  fill = president)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~president,
             scales = "free_y")   # y축 통일하지 않음

# 특정 단어 제거하고 막대 그래프 만들기
top10 <- frequency %>%
  filter(word != "국민") %>%
  group_by(president) %>%
  slice_max(n, n = 10, with_ties = F)

top10

ggplot(top10, aes(x = reorder(word, n),
                  y = n,
                  fill = president)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~president, scales="free_y")

# 축 정렬하기 - reorder_within()
ggplot(top10, aes(x = reorder_within(word, n, president),
                  y = n,
                  fill = president)) +
  geom_col() +
  coord_flip() + 
  facet_wrap(~ president, scales = "free_y")

# 변수 항목 제거하기 - scale_x_reordered()
ggplot(top10, aes(x = reorder_within(word, n, president),
                  y = n,
                  fill = president)) + 
  geom_col() +
  coord_flip() +
  facet_wrap(~president, scales="free_y") +
  scale_x_reordered() +
  labs(x = NULL) +  # x축 이름 삭제
  theme(text = element_text(family="nanumgothic"))    # 폰트

# long_form > wide_form 변환
df_long <- frequency %>%
  group_by(president) %>%
  slice_max(n, n = 10) %>%
  filter(word %in% c("국민", "우리", "정치", "행복"))

df_long

install.packages("tidyr")
library(tidyr)

df_wide <- df_long %>%
  pivot_wider(names_from = president,
              values_from = n)
df_wide


# NA를 0으로 바꾸기
df_wide <- df_long %>%
  pivot_wider(names_from = president,
              values_from = n,
              values_fill = list(n=0))
df_wide

# 연설문 단어 빈도를 wide form으로 변환하기
frequency_wide <- frequency %>%
  pivot_wider(names_from = president,
              values_from = n,
              values_fill = list(n = 0))
frequency_wide

# 오즈비 구하기
frequency_wide <- frequency_wide %>%
  mutate(ratio_moon = ((moon + 1) / (sum(moon + 1))), # moon에서 단어의 비중
         ratio_park = ((park + 1) / (sum(park + 1)))) # park에서 단어의 비중
frequency_wide

# 오즈비 변수 추가하기
frequency_wide <- frequency_wide %>%
  mutate(odds_ratio = ratio_moon / ratio_park)

frequency_wide %>% arrange(-odds_ratio)

frequency_wide %>% arrange(odds_ratio)

# 오즈비 간단히 구하기
frequency_wide <- frequency_wide %>%
  mutate(ratio_moon = ((moon + 1) / (sum(moon + 1))),
         ratio_park = ((park + 1) / (sum(park + 1))),
         odds_ratio = ratio_moon / ratio_park)

# 상대적으로 중요한 단어 추출하기
top10 <- frequency_wide %>%
  filter(rank(odds_ratio) <= 10 | rank(-odds_ratio) <= 10)

top10 %>%
  arrange(-odds_ratio) %>%
  print(n = Inf)