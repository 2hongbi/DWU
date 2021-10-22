### 1번문제
A <- matrix(c(6,2,2,2,2,0,2,0,2), byrow=TRUE, nr=3)

E1 <- diag(3)
E1[2, 1] <- -1/3
E1A <- E1 %*% A

cbind(E1, A, E1A)


E2 <- diag(3)
E2[3, 1] <- -1/3
E2E1A <- E2 %*% E1A

cbind(E2, E1A, E2E1A)

E3 <- diag(3)
E3[3, 2] <- 0.5
E3E2E1A <- E3 %*% E2E1A

cbind(E3, E2E1A, E3E2E1A)

L <- solve(E1) %% solve(E2) %% solve(E3)
U <- E3E2E1A
LU <- L %*% U

cbind(L, U, LU)

solve(A, c(0,0,0))

det(A)

invA <- solve(A)
invA

### 2번 문제 -> 손으로 풀기 skip

### 3번 문제
X <- matrix(c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0),
            byrow=TRUE, nr=6)
X

y <- c(-1, 1, 9, 11, 19, 21)
cbind(X, y)

# 연립방정식 행렬 표현
XTX <- t(X) %*% X
XTy <- t(X) %*% y
NE <- cbind(XTX, XTy)
NE

# 베타 = b


# Xb 구하기
b <- solve(XTX, XTy)
b

# e = y-\hat , 그 길이의 제곱
yh <- X %*% b
e <- y - yh
round(data.frame(y=y, yh=yh, e), 4)

sum(e*e)

# H = X(XTX)-1XT 구함
invXTX <- solve(XTX)


# H^2 = H 확인
H <- X %*% invXTX %*% t(X)
round(H, S)

round(H%*%H, S)


# tr(H) 구하기
sum(diag(H))

### 4번 문제

# 원자료를 X에 저장
X <- matrix(c(2, 3, 3, 2, 7, 1), byrow=TRUE, nr=3)
print(X)

# X의 중심화 행렬 :  평균 수정된 행렬
## Scale 사용
scale(X, center=TRUE, scale=FALSE)

## 행렬연산
XT <- t(X)
XT

S <- 0.5*(XT%*%X)
S

# X의 표준화 행렬
## scale 사용
scale(X, center=TRUE, scale=TRUE)

## 행렬연산
ST <- t(S)
R <- 0.5 * (ST %*% S)
R

# X의 교차제곱합 행렬
t(Z)%*%Z

# X의 수정 교차제곱합 행렬

# S(X의 공분산 행렬)
## cov 사용
S = cov(X)
S

## 행렬연산
I3 <- diag(3)
J3 <- matrix(1/nrow(X), nr=nrow(X), nc=nrow(X))
C3 <- I3-J3
cbind(I3, J3, C3)

# R(X의 상관행렬)
# cor 사용
R = cor(X)
R

# 행렬연산



### 5번 문제
library(tidyverse)
# htwtage4.csv 읽기
X <- read_csv('C:/Users/foxgi/OneDrive/project/DWU/multivariate/htwtage4.csv', col_names = TRUE)
X

# S 공분산 행렬
## 분광분해
S = cov(X)
eigS <- eigen(S)
V <- eigS$vector
L <- eigS$value
cbind(V, diag(L), t(V), V%*%diag(L)%*%t(V))

## S1, S2 등으로 복원되는지 확인하시오
S1 <- L[1]*V[,1]%*%t(V[,1])
S2 <- L[2]*V[,2]%*%t(V[,2])
cbind(S, S1, S1+S2)

# R 상관행렬
## 분광분해
R = cor(X)
eigR <- eigen(R)
V <- eigR$vector
L <- eigR$value
cbind(V, diag(L), t(V), V%*%diag(L)%*%t(V))

## R1, R2 등으로 복원 확인
R1 <- L[1]*V[,1]%*%t(V[,1])
R2 <- L[2]*V[,2]%*%t(V[,2])
cbind(R, R1, R1+R2)


# CX 중심화된 X
## SVD 분해
I3 <- diag(4)
J3 <- matrix(1/nrow(X), nr=nrow(X), nc=nrow(X))
C3 <- I3 - J3
cbind(I3, J3, C3)

CX <- C3%*%X
svdCX <- svd(CX)
## CX1, CX2 복원 확인

# X 원자료
## svd 분해
svdX <- svd(X)
svdX
## X1, X2 복원 확인





### 6번 문제
X <- matrix(c(5, 1, 2, 3, 2, 1, 2, 1, 5, 1, 4, 2, 2, 2, 3, 3, 3, 1, 4, 2, 3),
            byrow=TRUE, nr=7)
X

# fmsb 이용
library(fmsb)
library(RColorBrewer)

scale(X, center=TRUE, scale=TRUE)


