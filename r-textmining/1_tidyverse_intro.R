# 1. tidyverse 패키지

## tidyverse 패키지 실행
install.packages("tidyverse")

library(tidyverse)


# pipe(%>%)
plot(diff(log(sample(rnorm(n=10000, mean=10, sd=1), size=100, replace=FALSE))),
     col='red', type="l")

rnorm(n=10000, mean=10, sd=1) %>%
  sample(size=100, replace=FALSE) %>%
  log %>%
  diff %>%
  plot(col='red', type='l')



## tidyverse를 이용한 데이터 변환 p.28
library(ggplot2)
data(mpg)
glimpse(mpg)

mpg
 
# 예제 데이터 : mpg(ggplot2)
mpg %>% str()

## 예제 데이터 : menu data, source: kaggle.com p.31
library(readr)
menu <- read_csv("/Users/isoyeon/Documents/dwu/DWU/r-textmining/Data/menu.csv")
menu

# 데이터 선택
## summary(mpg$cty)
mpg %>% with(summary(cty))

# filter(mpg, cty > 17) 와 동일함, 사내 연비가 17 이상인 경우
mpg %>% filter(cty > 17)

# filter(mpg, cyl %in% c(4, 6))
mpg %>% filter(cyl %in% c(4, 6))

# filter(mpg, class=="suv")와 동일, 차의 형태가 suv인 경우
mpg %>% filter(class == "suv")

# Hyundal 차이면서도 고속도로 연비가 25 이상
mpg %>% filter(manufacturer == "hyundal" & hwy < 25)

