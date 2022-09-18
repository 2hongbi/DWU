install.packages("rJava")
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

install.packages("widyr")

raw_data <- read.xlsx("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/final_data.xlsx", 1) 
raw_data

raw_data <- as_tibble(raw_data)
raw_data

names(raw_data)
names(raw_data) = c("작성자", "작성일", "포스팅수", "탑승구간",
                                "클래스", "여행날짜", "작성제목", "댓글내용")
raw_data

# 댓글 내용 불러오기
data_comment <- raw_data %>%
  mutate('댓글내용'=str_replace_all(댓글내용, "[^가-힣]", " "),
         '댓글내용'=str_squish(댓글내용),
         id = row_number())
data_comment

# Q1-1. 명사, 형용사, 동사를 토큰으로 선택
# 명사 추출
extract_noun <- data_comment %>%
  unnest_tokens(input=댓글내용,
                output=word,
                token=extractNoun,
                drop=F) %>%
  filter(str_length(word) > 2) %>% print()

# 품사 기준 토큰화
comment_pos <- data_comment %>%
  unnest_tokens(input=댓글내용,
                output=word,
                token=SimplePos22,
                drop=F)
comment_pos

# 명사, 형용사, 동사 추출
comment <- comment_pos %>%
  separate_rows(word, sep = "[+]") %>%
  filter(str_detect(word, "n|/pv|/pa")) %>%
  mutate(word = ifelse(str_detect(word, "/pv|/pa"),
                       str_replace(word, "/.*$", "다"),
                       str_remove(word, "/.*$"))) %>%
  filter(str_count(word) >= 2) %>%
  arrange(id)
comment

# 빈도 높은 단어 제거
count_word <- comment %>%
  add_count(word) %>%
  filter(n <= 200) %>%
  select(-n)

count_word

# 불용어, 유의어 확인
count_word %>%
  count(word, sort = T) %>%
  print(n=400)

# 불용어 목록 만들기
stopword <- c("정도", "그렇다", "번째", "만큼", "였다")

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
                       "한국으로" = "한국"))
count_word

# Q2. 워드 클라우드
# 워드 클라우드를 그리기 위해 상위 100개만 뽑아오기
top100 <- count_word %>%
  count(word, sort = T) %>%
  filter(str_count(word) >= 2) %>%
  head(100)
top100

# mac 한글 꺠짐
theme_set(theme_gray(base_family = 'NanumGothic'))

# 한글깨짐으로 인해 wordcloud로 대체
wordcloud(words = top100$word,  # 단어들
          freq = top100$n, # 단어들의 빈도
          min.freq = 3,
          random.order = F, # 단어의 출력 위치, 빈도가 클수록 중앙에 위치
          rot.per = .1,  # 90도 회전 단어 비율
          scale = c(4, 0.3),  # 단어 폰트 크기
          colors = brewer.pal(8, "Dark2"),  # 단어 색
          family="AppleGothic")

# Q3. 연결중심성 크기 표시, 단어 네트워크 그림

# 형태소 분석기를 이용해 품사 기준으로 토큰화
comment_pos <- data_comment %>%
  unnest_tokens(input = 댓글내용,
                output = word,
                token = SimplePos22,
                drop = F) %>%
  select(word, 댓글내용)
comment_pos

# 품사별로 행 분리
separate_pos <- comment_pos %>%  # 위에서 정의했던 comment_pos 사용
  separate_rows(word, sep = "[+]") %>%  # sep에 입력한 정규 표현식에 따라 텍스트를 여러 행으로 나누는 함수
  select(word, 댓글내용) # 원하는 컬럼 선택

separate_pos


# 명사 추출하기- str_detect로 명사 추출 후, str_remove()를 이용해 태그 제거
noun <- separate_pos %>%
  filter(str_detect(word, "/n")) %>%
  mutate(word = str_remove(word, "/.*$")) %>%
  select(word, 댓글내용)

noun

# count() 이용해 댓글에 어떤 명사가 많이 사용되었는지 확인
noun %>%
  count(word, sort = T)


# 동사, 형용사 추출하기
pvpa <- separate_pos %>%
  filter(str_detect(word, "/pv|/pa")) %>%   # "/pv", "/pa" 추출
  mutate(word = str_replace(word, "/.*$", "다"))   # "/"로 시작 문자를 "다"로 바꾸기

pvpa %>%
  select(word, 댓글내용)

pvpa %>%
  count(word, sort = T)


# 품사 결합
comment <- bind_rows(noun, pvpa) %>%
  filter(str_count(word) >= 2) %>%
  

comment

# 명사 추출하기
noun <- separate_pos %>%
  filter(str_detect(word, "/n")) %>%
  mutate(word = str_remove(word, "/.*$"))
noun %>%
  select(word, 댓글내용)


# 명사 빈도 구하기
noun %>%
  count(word, sort = T)

pvpa <- comment_pos %>%
  filter(str_detect(word, "/pv|/pa")) %>% # "/pv", "/pa" 추출
  mutate(word = str_replace(word, "/.*$", "다")) # "/"로 시작 문자를 "다"로 바꾸기
pvpa %>%
  select(word, 댓글내용)

# 품사 결합
comment <- bind_rows(noun, pvpa) %>%
  filter(str_count(word) >= 2) %>%
  mutate(id = row_number())
comment

# 단어 동시 출현 빈도 구하기 - pairwise_count()
pair <- comment %>%
  pairwise_count(item = word,  # 단어
                 feature = id,  # 텍스트 구분 기준
                 sort = T)   # 내림차순으로 출력 결과 정렬
pair

# 네트워크 그래프 만들기 - as_tbl_graph()
graph_comment <- pair %>%
  filter(n >= 25)