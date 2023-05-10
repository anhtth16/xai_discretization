# Calculate inconsistency rate after discretization
# Source: R documentation, library discretization
# Update: 13.10.22, by Sam
library(readr)
library(discretization)

#--- CHECK: pima
#----
## ChiMerge
pima_cm6 <- read_csv('chim_pima_6int.csv')
incon_pima_cm6 <- incon(pima_cm6) # 0.05797101

pima_cm8 <- read_csv('chim_pima_8int.csv')
incon_pima_cm8 <- incon(pima_cm8) #  0.03333333

pima_cm10 <- read_csv('chim_pima_10int.csv')
incon_pima_cm10 <- incon(pima_cm10) # 0.02753623

pima_cm15 <- read_csv('chim_pima_15int.csv')
incon_pima_cm15 <- incon(pima_cm15) # 0.007246377


## Decistion Tree Discretizer

pima_dt2 <- read_csv('DT_small_discretized_pima.csv') # max_depth = 2
incon_pima_dt2 <- incon(pima_dt2) # 0.007246377

pima_dt3 <- read_csv('DT_medium_discretized_pima.csv') # max_depth = 3
incon_pima_dt3 <- incon(pima_dt3) # 0.001449275

pima_dt4 <- read_csv('DT_large_discretized_pima.csv') # max_depth = 4
incon_pima_dt4 <- incon(pima_dt4)

pima_dt5 <- read_csv('DT_verylarge_discretized_pima.csv') # max_depth = 5
incon_pima_dt5 <- incon(pima_dt5)



