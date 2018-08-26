library('Matrix')
library(ArmaUtils)

randomWalkMatrix <- function(n, steps){
  M <-matrix(0.0,n,n)
  x <- n/2
  y <- n/2
  for (i in 1:steps)
  {
    randomNumber <- sample.int(4, 1)
    if (randomNumber == 1)
    {
      x <- x + 1
      if (x > n)
        x <- n
      M[x,y] <- M[x,y] + 1.0
    }
    else if (randomNumber == 2)
    {
      y <- y + 1
      if (y > n)
        y <- n
      M[x,y] <- M[x,y] + 1.0
    }
    else if (randomNumber == 3)
    {
      x <- x - 1
      if (x < 1)
        x <- 1
      M[x,y] <- M[x,y] + 1.0
    }
    else
    {
      y <- y - 1
      if (y < 1)
        y <- 1
      M[x,y] <- M[x,y] + 1.0
    }
  }
  #Normalize the rows
  M <- t(t(M)/rowSums(M))
  return(M)
}

hittingTime <- function(P,Q)
{
  d1=NROW(P)
  
  d=d1*d1
  
  
  diagonalelements <- c()
  
  diag=0
  for (i in 1:d)
          {
           if (i==(diag+(d1*diag)+1) )
             {
              diagonalelements <- c(diagonalelements, i)
              diag <- diag + 1
             }           
  }     
  
  
  
  
  ## Target states
  hittingset <- diagonalelements
  
  
  one=array(1,c(1,d))
  
  one[hittingset] = 0
  
  mask = array(0,c(1,d))
  
  for (i in 1:d)
          {
            if (i %in% hittingset) {
              mask[i]=1
              }
          }    
  
  
  k1 = array(0,c(1,d))
  k2=armakron(y=k1,list(Q,P)) 
  
  
  k3 = one + t(k2)
   
  k3[mask] = 0
  while(norm(k1-k3)>1e-1){
          k1=k3
          k2=armakron(y=k1,list(Q,P))
          k3 = one + t(k2)
          k3=k3*one  
  }
  
}

size = 2
max_time = 60.0
last_spent_time = 0
time <- c()
sizes <- c()
#run test
while(1)
{
  start_time <- proc.time()
  P <- randomWalkMatrix(size, 100000)
  Q <- randomWalkMatrix(size, 100000)
  hittingTime(P,Q)
  last_spent_time <- proc.time() - start_time
  
  time <- c(time,last_spent_time[3])
  sizes <- c(sizes, size)
  
  if(as.numeric(last_spent_time[3]) > max_time)
    break
  size <- size * 2
  
  print(size)
}


# Create Bar Plot 
barplot(time,names.arg=sizes,xlab="N",ylab="Time to Compute Hitting Time",col="blue",
        main="Hitting Time Bar Graph",border="black")

