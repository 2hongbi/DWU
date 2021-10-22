### (B.1) 행렬의 생성

a <- -4.7 # 스칼라 생성
b <- c(1, 2, 3, 4, 5, 6, 7, 8, 9) # 벡터 생성 : c 함수
M <- matrix(b, nrow=3, ncol=3)  # 벡터를 행렬로 바꾸기: matrix 함수
print(M)

N <- matrix(b, nrow=3, ncol=3, byrow=TRUE)  # 행 기준으로 바꾸기
print(N)

v <- c(M)   # 행렬을 벡터로 바꾸기: c 함수
print(v)

student <- read.csv("C:/Users/foxgi/OneDrive/project/DWU/multivariate/data/student.csv",
                    header=TRUE)
student.X <- as.matrix(student[-1])  # 데이터 프레임을 행렬로 바꾸기
print(student.X)

dim(student.X)  # 행렬의 차수: dim 함수

attributes(student.X)  # 행렬의 차수, 행 이름, 열 이름 : attributes 함수


### (B.2) 기본적인 행렬 연산

A <- matrix(c(2, 5, 7, 9, 3, 4), nrow=3, ncol=2)
print(A)


B <- matrix(c(1, 2, 3, 4, 5, 6), nrow=3, ncol=2)
print(B)

B.t <- t(B)  # 행렬의 전치 : t 함수
print(B.t)

C <- A + B  # 두 행렬의 합
print(C)

D <- A - B   # 두 행렬의 차
print(D)

E <- 4*A   # 스칼라와 행렬의 곱
print(E)

F <- A %*%B.t   # 두 행렬의 곱 : %*% 연산자
print(F)

x <- c(1, 2, 3, 4, 5)
y <- c(3, 2, 4, 1, 6)
xy <- t(x)%*%y  # 두 벡터의 스칼라곱
print(xy)

xy.u <- sqrt(t(x-y)%*%(x-y))  # 두 벡터 사이의 유클리드 거리
print(xy.u)

G <- A*B  # 두 행렬의 동일한 위치의 원소들의 산술 곱
print(G)

L <- cbind(A, B)  # 두 행렬의 가로 결합 : cbind 함수
print(L)

R <- rbind(A, B)  # 두 행렬의 세로 결합 : rbind 함수
print(R)

### (B.3) 특수한 행렬들의 생성

A <- diag(3)    # 단위 행렬 생성 :diag 함수
print(A)

v <- c(1, 2, 3)
v.D <- diag(v)    # 벡터를 대각행렬로 바꾸기 :diag 함수
print(v.D)

M <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9), nrow=3)    # M <- matrix(1:9, nrow=3)
M.v <- diag(M)    # 행렬의 대각원소로 벡터 생성 : diag 함수
print(M.v)

M.D <- diag(diag(M))    # 행렬의 대각원소로 대각행렬 생성 :diag 함수
print(M.D)

one <- rep(1, 9)    # 모든 원소가 동일한 벡터 : rep 함수
print(one)

J3 <- matrix(rep(1, 9), nrow=3, ncol=3)    # 모든 원소가 동일한 벡터
print(J3)

M.L <- M[c(1, 3),]   # 행 추출, 참조: M[1:3,]
print(M.L)

M.R <- M[, c(1, 3)]   # 열 추출, 참조: M[, 1:3]
print(M.R)

M.K <- M[c(1,3), c(1,3)]   # 행/열 추출
print(M.K)
