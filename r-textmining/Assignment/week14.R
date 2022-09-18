# 과제10: speeches_roh.csv에는 노무현 전 대통령의 연설문 780개가 들어있습니다. 
# speeches_roh.csv를 이용해 문제를 해결하시오.
library(readr)
library(dplyr)
raw_speeches <- read_csv("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/speeches_roh.csv")
glimpse(raw_speeches)

# Q1. speeches_roh.csv를 불러온 다음 연설문이 들어있는 content를 문장 기준으로 토큰화하시오.
library(dplyr)
library(tidytext)
speeches <- raw_speeches %>%
  unnest_tokens(input=content,
                output=sentence,
                token="sentences",
                drop=F)
speeches

# Q2. 문장을 분석에 적합하게 전처리한 다음 명사를 추출하시오.
# 전처리
library(stringr)
speeches <- speeches %>%
  mutate(sentence = str_replace_all(sentence, "[^가-힣]", " "),
         sentence = str_squish(sentence))

# 명사 추출
library(tidytext)
library(KoNLP)
library(stringr)
nouns_speeches <- speeches %>%
  unnest_tokens(input = sentence,
                output = word,
                token = extractNoun,
                drop = F) %>%
  filter(str_count(word) > 1)
nouns_speeches

# Q3. 연설문 내 중복 단어를 제거하고 빈도가 100회 이하인 단어를 추출하시오.
nouns_speeches <- nouns_speeches %>%
  group_by(id) %>%
  distinct(word, .keep_all = T) %>%
  ungroup()

nouns_speeches

# 100회 이하 찾기
nouns_speeches <- nouns_speeches %>%
  add_count(word) %>%
  filter(n <= 100) %>%
  select(-n)

nouns_speeches

# Q4. 추출한 단어에서 다음의 불용어를 제거하시오.
stopword <- c("들이", "하다", "하게", "하면", "해서", "이번", "하네",
              "해요", "이것", "니들", "하기", "하지", "한거", "해주",
              "그것", "어디", "여기", "까지", "이거", "하신", "만큼")
nouns_speeches <- nouns_speeches %>%
  filter(!word %in% stopword)

nouns_speeches

# Q5. 연설문 별 단어 빈도를 구한 다음 DTM을 만드시오.
library(tm)
count_word <- nouns_speeches %>%
  count(id, word, sort = T)
count_word

dtm_word <- count_word %>%
  cast_dtm(document = id, term = word, value = n)
dtm_word

# Q6. 토픽 수를 2~20개로 바꿔가며 LDA 모델을 만든 다음 최적 토픽 수를 구하시오.
install.packages("ldatuning")
library(ldatuning)

models <- FindTopicsNumber(dtm = dtm_word,
                           topics = 2:20,
                           return_models = T,
                           control = list(seed = 1234))
FindTopicsNumber_plot(models)

# Q7. 토픽 수가 9개인 LDA 모델을 추출하세요.

# Q8. LDA 모델의 beta를 이용해 각 토픽에 등장할 확률이 높은 상위 10개 단어를 추출한 다음 토픽 별 주요 단어를 나타낸 막대 그래프를 만드시오.

# Q9. LDA 모델의 gamma를 이용해 연설문 원문을 확률이 가장 높은 토픽으로 분류하시오.

# Q10. 토픽 별 문서 수를 출력하시오.

# Q11. 문서가 가장 많은 토픽의 연설문을 gamma가 높은 순으로 출력하고 내용이 비슷한지 살펴보시오.

