# Calculate inconsistency rate after discretization
# Source: R documentation, library discretization
# Update: 13.10.22, by Sam
library(readr)
library(discretization)

#--- CHECK: australia
#----
## EWD
aus_ewd1 <- read_csv('aus_ewd1.csv')
incon_aus_ewd1 <- incon(aus_ewd1) #

aus_ewd2 <- read_csv('aus_ewd2.csv')
incon_aus_ewd2 <- incon(aus_ewd2) #

aus_ewd3 <- read_csv('aus_ewd3.csv')
incon_aus_ewd3 <- incon(aus_ewd3) # 

## EFD
aus_efd1 <- read_csv('aus_efd1.csv') # 
incon_aus_efd1 <- incon(aus_efd1)

aus_efd2 <- read_csv('aus_efd2.csv')
incon_aus_efd2 <- incon(aus_efd2) # 

aus_efd3 <- read_csv('aus_efd3.csv')
incon_aus_efd3 <- incon(aus_efd3) # 

## FFD
aus_ffd1 <- read_csv('aus_ffd1.csv') # 
incon_aus_ffd1 <- incon(aus_ffd1)

aus_ffd2 <- read_csv('aus_ffd2.csv') # 
incon_aus_ffd2 <- incon(aus_ffd2)

aus_ffd3 <- read_csv('aus_ffd3.csv')
incon_aus_ffd3 <- incon(aus_ffd3) # 

aus_ffd4 <- read_csv('aus_ffd4.csv')
incon_aus_ffd4 <- incon(aus_ffd4) # 
