# 연습문제 1 : 기존 baseR 방식과 tidyR 방식을 이용하여 다음을 구하시오.

library(ggplot2)
data(mpg)
glimpse(mpg)

mpg

## hwy가 10 이상인 것들만 추려내시오.
# baseR
filter(mpg, hwy > 10)

# tidyR
mpg %>% filter(hwy > 10)

## year가 2000년 이후인 것만 추려내시오
# baseR
filter(mpg, year >= 2000)

# tidyR
mpg %>% filter(year >= 2000)

## cty가 10 미만이고, year가 2000년 미만인 것들만 추려내시오.(Et를 활용할 것)
# baseR
filter(mpg, cty < 10 & year < 2000)

# tidyR
mpg %>% filter(cty < 10 & year < 2000)

# displ이 1.8인 경우만 추려내시오
#baseR
filter(mpg, displ == 1.8)

# tidyR
mpg %>% filter(displ == 1.8)

## displ이 2.0이고 cyl이 6.8인 경우만 추려내시오
# baseR
filter(mpg, displ == 2.0 & cyl >= 6.8)

# tidyR
mpg %>% filter(displ == 2.0 & cyl >= 6.8)
