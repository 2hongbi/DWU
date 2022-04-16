# 문자열 벡터 읽기
moon <- readLines("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/문재인출마선언문.txt", encoding="UTF-8")
moon

head(moon)

install.packages("stringr")
library(stringr)

moon1 <- moon %>% str_replace_all("[^가-힣]", " ")
moon1

# 연설문 연속된 공백 제거
moon1 <- moon1 %>% str_squish()
moon1

# 문자열 벡터 tibble 구조로 바꾸기
library(dplyr)
moon1 <- as_tibble(moon1)
moon1


write.table(moon1, file='moon_20191775_이소연.csv')