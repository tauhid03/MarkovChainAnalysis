library('Matrix')
library(ArmaUtils)

randomWalkMatrix <- function(n, steps){
  M <-matrix(0.0,n,n)
  #Add artifical data to the matrix
  for (row in 1:nrow(M))
  {
    #M goes from 1 to n
    M[row,row] <- M[row,row] + 1
    if(row > 1)
    {
      M[row-1, row-1] <- M[row-1, row-1] + 1
      M[row-1, row] <- M[row-1, row] + 1
      M[row, row-1] <- M[row, row-1] + 1
    }
    if(row < n)
    {
      M[row+1, row+1] <- M[row+1, row+1] + 1
      M[row+1, row] <- M[row+1, row] + 1
      M[row, row+1] <- M[row, row+1] + 1
    }
  }
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
  i = 0
  foo <- tryCatch(
  {
  while(norm(k1-k3)>1e-1){
          i = i +1
          k1=k3
          k2=armakron(y=k1,list(Q,P))
          k3 = one + t(k2)
          k3=k3*one  
  }
  },
  finally={

   # message("norm(k1-k3) =")
  #  message(norm(k1-k3))
  #  message("k1 =")
  #  message(k1)
  #  message("k3 =")
  #  message(k3)
  }
  )
  
}

size = 2
max_time = 10
last_spent_time = 0
time <- c()
sizes <- c()
#run test
while(1)
{
  print("New test size is")
  print(size)
  start_time <- proc.time()
  P <- randomWalkMatrix(size, 100000)
  Q <- randomWalkMatrix(size, 100000)
  HT <- hittingTime(P,Q)
  last_spent_time <- proc.time() - start_time
  time <- c(time,last_spent_time[3])
  sizes <- c(sizes, size)
  
  if(as.numeric(last_spent_time[3]) > max_time)
    break
  size <- size * 2

}

# Create Bar Plot 
barplot(time,names.arg=sizes,xlab="N",ylab="Time to Compute Hitting Time (Seconds)",col="blue",
        main="Hitting Time Bar Graph",border="black")

# Create Bar Plot 
barplot(time,names.arg=sizes,xlab="N",ylab="Time to Compute Hitting Time",col="blue",
main="Hitting Time Bar Graph",border="black")

