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