# Calculate inconsistency rate after discretization
# Source: R documentation, library discretization
# Update: 13.10.22, by Sam
library(readr)
library(discretization)

#--- CHECK: australia
#----
## ChiMerge
aus_cm6 <- read_csv('chim_aus_6int.csv')
incon_aus_cm6 <- incon(aus_cm6) # 0.05797101

aus_cm8 <- read_csv('chim_aus_8int.csv')
incon_aus_cm8 <- incon(aus_cm8) #  0.03333333

aus_cm10 <- read_csv('chim_aus_10int.csv')
incon_aus_cm10 <- incon(aus_cm10) # 0.02753623

aus_cm15 <- read_csv('chim_aus_15int.csv')
incon_aus_cm15 <- incon(aus_cm15) # 0.007246377


## Decistion Tree Discretizer

aus_dt2 <- read_csv('DT_small_discretized_aus.csv') # max_depth = 2
incon_aus_dt2 <- incon(aus_dt2) # 0.007246377

aus_dt3 <- read_csv('DT_medium_discretized_aus.csv') # max_depth = 3
incon_aus_dt3 <- incon(aus_dt3) # 0.001449275

aus_dt4 <- read_csv('DT_large_discretized_aus.csv') # max_depth = 4
incon_aus_dt4 <- incon(aus_dt4)

aus_dt5 <- read_csv('DT_verylarge_discretized_aus.csv') # max_depth = 5
incon_aus_dt5 <- incon(aus_dt5)



