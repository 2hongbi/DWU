install.packages("multilinguer")
library(multilinguer)
install_jdk()

install.packages(c("hash", "tau", "Sejong", "RSQLite", "devtools"), type = "binary")

install.packages("remotes")
library(remotes)
install_github("haven-jeon/KoNLP", upgrade="never", INSTALL_opts=c("--no-multiarch"))
library(KoNLP)

# 형태소 사전 설정하기
useNIADic()