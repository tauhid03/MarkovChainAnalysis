library('Matrix')
library(ArmaUtils)

P <- matrix(c(0.2, 0.8,0.4, 0.6), nrow =2, ncol = 2)
Q <- matrix(c(0.3, 0.7,0.1, 0.9), nrow =2, ncol = 2)


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
            print(i)  
           }           
}     




## Target states
hittingset <- diagonalelements

print("Target States")
print(hittingset)

one=array(1,c(1,d))

one[hittingset] = 0

mask = array(0,c(1,d))

for (i in 1:d)
        {
          if (i %in% hittingset) {
            mask[i]=1
            }
         print(i)
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

print("Hitting Time")
print(k3)







