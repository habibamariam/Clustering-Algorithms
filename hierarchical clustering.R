dataset<-read.csv('Mall_Customers.csv')
X<- dataset[4:5]

#using dengrogram for optimal no of clusters
dendrogram=hclust(dist(X,method='euclidean'),method = 'ward.D')
plot(dendrogram,main=paste('dendrogram'),ylab = 'eucledean distance',xlab='customers')

#fitting the Hc in dataset

hc =hclust(dist(X,method='euclidean'),method = 'ward.D')
y_hc=cutree(hc,5)

set.seed(29)
hc<-hc(X,5,iter.max=300,nstart=10)

#visualizing the clusters
clusplot(X,y_hc,lines=0,shade=TRUE,color=TRUE,labels=2,plotchar=FALSE,span= TRUE,main=paste('Clusters of clients'),xlab='annual income',ylab='spending score')