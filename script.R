rm(list=ls())

library('simone')
library('mlbench')
library('e1071')
library('MASS')
library('AtmRay')
library('varhandle')
library('rgl')
library('clusterSim')

#------------------Função de amostragem dos dados
splitDataTrainAndTest<-function(xc1,xc2,proportion)
{
  
  #Definindo os limites
  c1l1<-1:(proportion[1]*nrow(xc1))
  c1l2<-(c1l1[length(c1l1)]+1):(sum(proportion[1:2])*nrow(xc1))
  
  c2l1<-1:(proportion[1]*nrow(xc2))
  c2l2<-(c2l1[length(c2l1)]+1):(sum(proportion[1:2])*nrow(xc2))
  
  #Separando treino e teste
  x1train <- xc1[c1l1,]
  x1validation <- xc1[c1l2,]
  
  x2train <- xc2[c2l1,]
  x2validation <- xc2[c2l2,]
  
  trainY<-matrix(-1,nrow=(nrow(x1train)+nrow(x2train)),ncol=1)
  trainY[1:nrow(x1train),]<-1
  trainX <- rbind(x1train,x2train)
  
  valY<-matrix(-1,nrow=(nrow(x1validation)+nrow(x2validation)),ncol=1)
  valY[1:nrow(x1validation),]<-1
  valX<-rbind(x1validation,x2validation)
  
  output<-list(trainX=trainX,trainY=trainY,testX=valX,testY=valY)
  
  return(output)
}


#BREAST CANCER
#----------------------------

data("BreastCancer")

BC <- data.matrix(BreastCancer)[,2:11]
BC[is.na(BC)]<-0

acc.can.mean <- matrix(0, nrow=10, ncol=2)
acc.bc.mean<-acc.can.mean
acc.can.sd<-acc.can.mean
acc.bc.sd<-acc.can.mean
for(it in 1:10){
acc.bc<-matrix(0,nrow=50,ncol=2)
for(i in 1:50)
{
  
  BC <- BC[sample(nrow(BC)),]
  
  X <- BC[,1:9]
  Y <- BC[,10]
  Y[which(Y=='2')] <- -1
  
  #Separando os dados de treino e teste
  split_data<-splitDataTrainAndTest(X[which(Y=='1'),],
                                    X[which(Y=='-1'),],c(0.8,0.2))
  trainX<-split_data[[1]]
  trainY<-split_data[[2]]
  testX<-split_data[[3]]
  testY<-split_data[[4]]
  
  #Principal components analysis
  trainX.means<-apply(trainX,2,mean)
  trainX_xm<-trainX-matrix(trainX.means,nrow=nrow(trainX),
                           ncol=ncol(trainX),byrow=TRUE)
  eigS<-eigen(cov(trainX_xm))
  trainX.PC<-trainX_xm%*%eigS[[2]]
  testX.PC<-(testX-matrix(trainX.means,nrow=nrow(testX),
                          ncol=ncol(testX),byrow=TRUE))%*%eigS[[2]]

  
  #Plotando resultado
  #Sys.sleep(2)
  plot(trainX.PC[trainY=='1',1],trainX.PC[trainY=='1',2],
       xlim=c(-25,15),ylim=c(-20,20),xlab='PCA1',ylab='PCA2',col='magenta',
       main = 'Breast Cancer Components')
  par(new=T)
  plot(trainX.PC[trainY=='-1',1],trainX.PC[trainY=='-1',2],
       xlim=c(-25,15),ylim=c(-20,20),xlab='',ylab='',col='blue')
  
  #Modelo de classificação
  svm.model.inspace<-svm(trainY ~ ., data=trainX,
                         cost=1,gamma=0.1)
  yhat.inspace<-predict(svm.model.inspace,testX)
  yhat.inspace[yhat.inspace>0]<-1
  yhat.inspace[yhat.inspace<0]<--1
  
  svm.model.pca<-svm(trainY ~ ., data=trainX.PC[,1:3],
                     cost=1,gamma=0.1)
  yhat.pca<-predict(svm.model.pca,testX.PC[,1:3])
  yhat.pca[yhat.pca>0]<-1
  yhat.pca[yhat.pca<0]<--1
  
  acc.bc[i,1]<-sum(diag(table(testY,yhat.inspace)))/sum(table(testY,yhat.inspace))
  acc.bc[i,2]<-sum(diag(table(testY,yhat.pca)))/sum(table(testY,yhat.pca))
}

plot(eigS[[1]]/sum(eigS[[1]])*100,type='b',col='red',
     xlab='PCA',ylab='Var_explained (%)',main='Breast Cancer - Variância explicada x Componente')

acc.bc.mean[it,]<-apply(acc.bc,2,mean)
acc.bc.sd[it,]<-apply(acc.bc,2,sd)



#-------------------------------------------------------------
#CANCER
data("cancer")

CAN<-data.matrix(cancer[[1]])
Y<-c()
Y[which(levels(cancer[[2]])[1]==cancer[[2]])]<-1
Y[which(levels(cancer[[2]])[2]==cancer[[2]])]<--1

CAN<-cbind(CAN,Y)

acc.can<-matrix(0,nrow=30,ncol=2)
for(i in 1:30)
{
  
  CAN<-CAN[sample(nrow(CAN)),]
  
  X <- CAN[,1:26]
  Y <- CAN[,27]
  
  #Separando os dados de treino e teste
  split_data<-splitDataTrainAndTest(X[which(Y=='1'),],
                                    X[which(Y=='-1'),],c(0.8,0.2))
  trainX<-split_data[[1]]
  trainY<-split_data[[2]]
  testX<-split_data[[3]]
  testY<-split_data[[4]]
  
  #Principal components analysis
  trainX.means<-apply(trainX,2,mean)
  trainX_xm<-trainX-matrix(trainX.means,nrow=nrow(trainX),
                           ncol=ncol(trainX),byrow=TRUE)
  eigS<-eigen(cov(trainX_xm))
  trainX.PC<-trainX_xm%*%eigS[[2]]
  testX.PC<-(testX-matrix(trainX.means,nrow=nrow(testX),
                          ncol=ncol(testX),byrow=TRUE))%*%eigS[[2]]
  
  
  #Plotando resultado
  #Sys.sleep(2)
  plot(trainX.PC[trainY=='1',1],trainX.PC[trainY=='1',2],
       xlim=c(-5,5),ylim=c(-5,7),xlab='',ylab='',col='magenta',
       main='Cancer Components')
  par(new=T)
  plot(trainX.PC[trainY=='-1',1],trainX.PC[trainY=='-1',2],
       xlim=c(-5,5),ylim=c(-5,7),xlab='PCA1',ylab='PCA2',col='blue')
  
  #Modelo de classificação
  svm.model.inspace<-svm(trainY ~ ., data=trainX,
                         cost=1,gamma=0.1)
  yhat.inspace<-predict(svm.model.inspace,testX)
  yhat.inspace[yhat.inspace>0]<-1
  yhat.inspace[yhat.inspace<0]<--1
  
  svm.model.pca<-svm(trainY ~ ., data=trainX.PC[,1:2],
                     cost=1,gamma=0.1)
  yhat.pca<-predict(svm.model.pca,testX.PC[,1:2])
  yhat.pca[yhat.pca>0]<-1
  yhat.pca[yhat.pca<0]<--1
  
  acc.can[i,1]<-sum(diag(table(testY,yhat.inspace)))/sum(table(testY,yhat.inspace))
  acc.can[i,2]<-sum(diag(table(testY,yhat.pca)))/sum(table(testY,yhat.pca))
}

plot(eigS[[1]]/sum(eigS[[1]])*100,type='b',col='red',
     xlab='PCA',ylab='Var_explained (%)',main='Cancer - Variância explicada x Componente')

acc.can.mean[it,]<-apply(acc.can,2,mean)
acc.can.sd[it,]<-apply(acc.can,2,sd)


#---------------------------------------------------------
data("USArrests")

USA<-data.matrix(USArrests)

USA.means<-apply(USA,2,mean)
aux<-matrix(0,nrow=nrow(USA),ncol=ncol(USA))
USA_xm<-USA-matrix(USA.means,nrow=nrow(USA),ncol=ncol(USA),byrow=TRUE)
eigS<-eigen(cov(USA_xm))
USA.PCA<-USA_xm%*%eigS[[2]]

plot(eigS[[1]]/sum(eigS[[1]])*100,type='b',xlab='PCA',ylab='Explained Var (%)',
     main = 'USArrests - Variância explicada x Componente')


print(it)
}
