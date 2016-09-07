kick.data <-read.csv("/Users/ramarvab/PycharmProjects/Apriori/training_preprocess.csv",header = TRUE,sep = ",")
kick.data <-na.omit(kick.data)
logMMR.kick <-(kick.data[,18:25])
kick.class <-kick.data[,2]
MMR.pca <-prcomp(logMMR.kick,center = TRUE, scale.=TRUE)
print(MMR.pca)
plot(ir.pca,type ="l",main ="PCA for MMR Features")

logVechicletime.kick <- (kick.data[,4:5])
vechicletime.pca <-prcomp(logVechicletime.kick,center = TRUE, scale.=TRUE)
print(vechicletime.pca)
plot(vechicletime.pca,type ="l",main ="PCA for vehicle purchase Features")
