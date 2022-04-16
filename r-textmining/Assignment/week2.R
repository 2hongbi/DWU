# 연습문제 2. 비행기 데이터 문제(mycflights13)

# nycflights13 패키지를 설치한다
install.packages("nycflights13")

# library(nycflights) 명령어로 패키지를 로드하고, flights 데이터를 불러온다
library(nycflights13)
data(flights)


# 비행달이 7, 8, 9월인 행만 추려내시오
flights %>% filter(month %in% c(7, 8, 9))

# 목적지(dest)가 “IAH”이거나 “HOU”인 행만 추려내시오
test <- flights %>% filter(dest %in% c("IAH", "HOU"))
test$dest

# 도착지연 시간(arr_delay)이 60분이고, 출발지연 시간(dep_delay)이 0분인 행만 추려내시오.
flights %>% filter(arr_delay == 60 & dep_delay == 0)


# Year, month, day열만 추려내시오.
flights %>% select(year, month, day)

# dep_time부터 arr_delay열까지 한꺼번에 추려내시오.
flights %>% select(dep_time:arr_delay)

# Year, month, day에 따른 dep_delay의 평균을 구하시오.(결측치도 처리할 것)
flights %>% filter(!is.na(dep_delay)) %>%
  group_by(year, month, day) %>%
  summarise(delay_mean = mean(dep_delay)) %>%
  ungroup()


# 목적지(dest)에 따른 dep_delay의 평균을 구해 내림차순으로 정리하시오
flights %>% group_by(dest) %>%
  summarise(delay_mean = mean(dep_delay)) %>%
  arrange(desc(delay_mean))
