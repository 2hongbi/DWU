sigma <- matrix(c(10000, 60, 60, 1), byrow=TRUE, nr=2)
sigma

# 상관행렬 로 구하기
P <- cov2cor(sigma)
P

# 공분산행렬에 근거한 주성분분석
eigen(sigma)
## 고유값은 1.6, 0.4, 고유벡터는

# 상관행렬에 근거한 주성분분석 -> 고유값과 고유벡터
eigen(P)

## 공분산행렬ㅇ에 기초하여 얻은 표본 주성분이 상관행렬 분포보다 훨씬 간단하다

  

 



proc.princomp <- function(x, cor=T){
  if(cor) S <- cor(x, use="complete")
  else S <- cov(x, use="complete")
  eigS <- eigen(S)
  dimnames(eigS$vectors) <- list(colnames(x), paste0("PC", 1:ncol(x)))
  list(coef=eigS$vectors, var=eigS$values, cor=cor)
}

summary.proc.princomp <- function(pc) {
  res <- cbind(sqrt(pc$var), pc$var, -c(diff(pc$var), NA), pc$var/sum(pc$var), cumsum(pc$var)/sum(pc$var))
  colnames(res) <- c('sd[PC]', 'EigenValue(V[PC])', 'Difference', 'Proportion', 'Cumulative')
  rownames(res) <- paste0("PC", 1:length(pc$var))
  res
}

predict.proc.princomp <- function(x, pc) {
  if(pc$cor) Z <- scale(x, center=TRUE, scale=TRUE)
  else Z <- scale(x, center=TRUE, scale=FALSE)
  pcscr <- data.frame(Z%*%pc$coef)
  pcscr
}

# 미국 범죄통계
library(tidyverse)
crime <- read_csv('C:/Users/foxgi/OneDrive/project/DWU/multivariate/linear_algebra/data/crime-kr.csv',
                  col_names = TRUE, locale=locale("ko", encoding="euc-kr"))
crime

## 6가지 범죄 사용하기 : 필요없는 컬럼 drop
crime <- crime[, -c(11:12)]
crime

# 상관행렬 R로 분석
crime.X <- crime[2:8]
crime.X

## 상관행렬 확인
R <- cor(crime.X) 
R

eig <- eigen(R)
V <- eig$vectors
L <- diag(eig$values)
eig
V
L

pc <- proc.princomp(crime.X)
summary.proc.princomp(pc)

## 분석
crime.cor.prcomp <- prcomp(crime.X, center=TRUE, scale=TRUE)
crime.cor.prcomp



#### 3. 주성분 개수를 정한 근거와 효과 : 주성분은 PC1과 PC2으로 정했습니다. (2개)
#### 이는 Kaiser의 규칙인 주성분이 상관행렬에 기초하고 있따면 상관행렬은 대각원소가 1이므로
#### 모든 주성분의 분산은 1이 되기에, 1보다 작은 고유값을 가지는 주성분은 원래 반응 변수 중
### 어느하나보다 작은 정보를 가지므로 보유할 가치가 없다고 판단했습니다. 
#### 때문에 분산값이 1이 넘는 PC1과 PC2만 선택했습니다.

#### 4. 주성분 1을 해석하고 이름 정하기
#### 일단 다 음수가 나오니까, 음수를 곱하고 보겠습니다. 
#### 주성분1이 높은 주는 7개의 범죄가 골고루 발생하는 경향을 보이고, 
#### 낮은 주는 7개의 범죄 발생건수가 다 낮다는 것을 확인할 수 있습니다.
#### 때문에 결과는 7가지 범죄 발생빈도 로 칭하겠습니다.


#### 5. 주성분 2를 해석하고 이름 정하기
#### 주성분 2가 높은 주는 살인, 성범죄, 폭행건수가 높은 경향을 보이고
#### 주성분 2가 낮은 주는 그 밖의 범죄들(강도, 절도, 차량절도, 가택침입절도)가 더 빈번히 일어나는 경향을 보였습니다.
#### 때문에 이는 3대범죄(살인, 성범죄, 폭행)와 그 외 범죄의 발생빈도 차이 라고 정하겠습니다.

predict.proc.princomp(crime.X, pc)

#### 6. 주성분 1의 최고 점수를 받은 주와 최저 점수를 받은 주 제시
#### 최고 점수를 받은 주는 North Dakota(3.96407765), 최저를 받은 주는 California(-4.28380373)
#### 7. 주성분 2의 최고 점수를 받은 주와 최저 점수를 받은 주 제시
#### 최고 점수를 받은 주는 New Mexico(9.507561e-01)과 최저 점수는 New Jersey(-9.642088e-01)


## 고유값 출력
crime.cor.prcomp$sdev^2

## 설명분산 요약
summary(crime.cor.prcomp)

crime.cor.score <- cbind(crime, crime.cor.prcomp$x[, 1:2])
crime.cor.score


### 8. 행렬도(biplot) 그리기
biplot(crime.cor.prcomp, cex=1)
abline(h=0, v=0, lty=2)

### 앞서 본 PC1이 다 음수였기 때문에, biplot의 PC1도 다 음수인 것을 확인할 수 있습니다.
### 또한 7개의 변수들이 모두 비슷한 길이를 갖고 있으므로 모두가 비슷한 정도의 
#### 높은 공통성을 갖고 있다고 판단됩니다. 
#### 때문에 사잇각으로 판단하면 살인은 폭행과 연관이 높고, 폭행은 강간과 연관이 높으며,
#### 차량절도는 절도와, 강도는 주택침입절도와 상대적으로 높은 연관성이 있었습니다.


# Oliveti 얼굴인식

oli <- read_csv('C:/Users/foxgi/OneDrive/project/DWU/multivariate/linear_algebra/data/olivetti_X.csv',
                  col_names = TRUE, locale=locale("ko", encoding="euc-kr"))
oli


# 공분산 행렬 S로 분석하기
oli.X <- oli[1:4096]
oli.X

S <- cov(oli.X)
S

