# Calculate inconsistency rate after discretization
# Source: R documentation, library discretization
# Update: 13.10.22, by Sam
library(readr)
library(discretization)

#--- CHECK: pima
#----
## EWD
pima_ewd1 <- read_csv('pima_ewd1.csv')
incon_pima_ewd1 <- incon(pima_ewd1) #

pima_ewd2 <- read_csv('pima_ewd2.csv')
incon_pima_ewd2 <- incon(pima_ewd2) #

pima_ewd3 <- read_csv('pima_ewd3.csv')
incon_pima_ewd3 <- incon(pima_ewd3) # 

## EFD
pima_efd1 <- read_csv('pima_efd1.csv') # 
incon_pima_efd1 <- incon(pima_efd1)

pima_efd2 <- read_csv('pima_efd2.csv')
incon_pima_efd2 <- incon(pima_efd2) # 

pima_efd3 <- read_csv('pima_efd3.csv')
incon_pima_efd3 <- incon(pima_efd3) # 

## FFD
pima_ffd1 <- read_csv('pima_ffd1.csv') # 
incon_pima_ffd1 <- incon(pima_ffd1)

pima_ffd2 <- read_csv('pima_ffd2.csv') # 
incon_pima_ffd2 <- incon(pima_ffd2)

pima_ffd3 <- read_csv('pima_ffd3.csv')
incon_pima_ffd3 <- incon(pima_ffd3) # 

pima_ffd4 <- read_csv('pima_ffd4.csv')
incon_pima_ffd4 <- incon(pima_ffd4) # 
