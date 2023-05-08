# Calculate inconsistency rate after discretization
# Project: Capita Selecta
# Datasets: musk
# Source: R documentation, library chiM
# Update: 23.03.23, by Sam

# Load library
library(readr)
library(discretization)

# # Sample code
# #--Discretization using the ChiMerge method
# data(adult)
# disc=chiM(adult,alpha=0.05)
# #--cut-points
# disc$cutp

#--CALCULATING INCONSISTENCY RATE
# The inconsistency rate of dataset is calculated as follows:
# (1) two instances are considered inconsistent if they match except for their class labels;
# (2) for all the matching instances (without considering their class labels),
# the inconsistency count is the number of the instances minus the largest number of instances of class labels;
# (3) the inconsistency rate is the sum of all the inconsistency counts
# divided by the total number of instances.

#--- CHECK: musk
## EWD
musk_ewd1 <- read_csv('musk_ewd1.csv')
incon_musk_ewd1 <- incon(musk_ewd1) # 

musk_ewd2 <- read_csv('musk_ewd2.csv')
incon_musk_ewd2 <- incon(musk_ewd2) # 

musk_ewd3 <- read_csv('musk_ewd3.csv')
incon_musk_ewd3 <- incon(musk_ewd3) # 

## EFD
musk_efd1 <- read_csv('musk_efd1.csv') # 
incon_musk_efd1 <- incon(musk_efd1)

musk_efd2 <- read_csv('musk_efd2.csv')
incon_musk_efd2 <- incon(musk_efd2) # 

musk_efd3 <- read_csv('musk_efd3.csv')
incon_musk_efd3 <- incon(musk_efd3) # 

## FFD
musk_ffd1 <- read_csv('musk_ffd1.csv') # 
incon_musk_ffd1 <- incon(musk_ffd1)

musk_ffd2 <- read_csv('musk_ffd2.csv') # 
incon_musk_ffd2 <- incon(musk_ffd2)

musk_ffd3 <- read_csv('musk_ffd3.csv')
incon_musk_ffd3 <- incon(musk_ffd3) # 

musk_ffd4 <- read_csv('musk_ffd4.csv')
incon_musk_ffd4 <- incon(musk_ffd4) # 
