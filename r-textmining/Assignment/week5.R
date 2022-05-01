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

# tibble에서 빈 문자열 ("") 지우기
moon1 <- moon1[!apply(moon1 == "", 1, all), ]
moon1

write.table(moon1, file='moon_20191775_이소연.csv')


string <- 'don22stat92동덕여대2022상월곡'
str_extract(string, "[a-z]{3,}")

string <- 'don22stat92동덕여대2022상월곡'
str_extract_all(string, "[^0-9가-힣]{5,}")

text <- c("ac", "abbc", "abc", "abcd", "abbdc", "abddc")
result <- grep(pattern="ab+c", x=text, value=TRUE)
result

r <- grep(pattern="ab?c", x=text, value=TRUE)
r

text <- c("good", "wasp", "sepcial", "statistic", "abbey load", "zoo", "batman24")

fruit <- c('apple', 'Apple', 'APPLE', 'banana', 'grape')
find <- grepl('pp', fruit)
find
sum(find)
g <- grepl(pattern="^[a-f]", x=text)
g
result <- sum(g)
result

phone <- c("042-868-9999", "02-3345-1234",
           "010-5651-7578", "016-123-4567",
           "063-123-5678", "070-5498-1904",
           "011-423-2912", "010-6745-2973")
g <- grep(pattern="^(01|04|02)\\d{0,}-\\d{3,4}-\\d{4}", x=phone, value=T)
g