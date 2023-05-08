# Calculate inconsistency rate after discretization
# Project: Capita Selecta
# Datasets: musk
# Discretization: Supervised (ChiMerge, DecisionTree)
# Source: R documentation, library chiM
# Update: 23.03.23, by Sam

# Load library
library(readr)
library(discretization)

# # Sample code
# #--Discretization using the ChiMerge method
# data(musk)
# disc=chiM(musk,alpha=0.05)
# #--cut-points
# disc$cutp

#--CALCULATING INCONSISTENCY RATE
# The inconsistency rate of dataset is calculated as follows:
# (1) two instances are considered inconsistent if they match except for their class labels;
# (2) for all the matching instances (without considering their class labels),
# the inconsistency count is the number of the instances minus the largest number of instances of class labels;
# (3) the inconsistency rate is the sum of all the inconsistency counts
# divided by the total number of instances.

# dismusk=disc$Disc.data
# icon_musk_sample <- incon(dismusk) #

#--- CHECK musk datasets
## ChiMerge
musk_cm1 <- read_csv('sc_cm_musk_6int.csv')
incon_musk_cm1 <- incon(musk_cm1) # 

musk_cm2 <- read_csv('sc_cm_musk_8int.csv')
incon_musk_cm2 <- incon(musk_cm2) # 

musk_cm3 <- read_csv('sc_cm_musk_10int.csv')
incon_musk_cm3 <- incon(musk_cm3) # 

musk_cm4 <- read_csv('sc_cm_musk_15int.csv')
incon_musk_cm4 <- incon(musk_cm4) # 

## DecisionTree
musk_dt1 <- read_csv('DT_small_discretized_musk.csv')
incon_musk_dt1 <- incon(musk_dt1) # 

musk_dt2 <- read_csv('DT_medium_discretized_musk.csv')
incon_musk_dt2 <- incon(musk_dt2) # 

musk_dt3 <- read_csv('DT_large_discretized_musk.csv')
incon_musk_dt3 <- incon(musk_dt3) # 

musk_dt4 <- read_csv('DT_verylarge_discretized_musk.csv')
incon_musk_dt4 <- incon(musk_dt4) #
