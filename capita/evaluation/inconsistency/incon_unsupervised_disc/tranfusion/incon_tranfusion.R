# Calculate inconsistency rate after discretization
# Project: Capita Selecta
# Datasets: tranfusion
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

#--- CHECK: tranfusion
## EWD
tranfusion_ewd1 <- read_csv('tranfusion_ewd1.csv')
incon_tranfusion_ewd1 <- incon(tranfusion_ewd1) # 

tranfusion_ewd2 <- read_csv('tranfusion_ewd2.csv')
incon_tranfusion_ewd2 <- incon(tranfusion_ewd2) # 

tranfusion_ewd3 <- read_csv('tranfusion_ewd3.csv')
incon_tranfusion_ewd3 <- incon(tranfusion_ewd3) # 

## EFD
tranfusion_efd1 <- read_csv('tranfusion_efd1.csv') #
incon_tranfusion_efd1 <- incon(tranfusion_efd1)

tranfusion_efd2 <- read_csv('tranfusion_efd2.csv') # 
incon_tranfusion_efd2 <- incon(tranfusion_efd2)

tranfusion_efd3 <- read_csv('tranfusion_efd3.csv') # 
incon_tranfusion_efd3 <- incon(tranfusion_efd3)

# FFD
tranfusion_ffd1 <- read_csv('tranfusion_ffd1.csv') # 
incon_tranfusion_ffd1 <- incon(tranfusion_ffd1)

tranfusion_ffd2 <- read_csv('tranfusion_ffd2.csv') # 
incon_tranfusion_ffd2 <- incon(tranfusion_ffd2)

tranfusion_ffd3 <- read_csv('tranfusion_ffd3.csv') # 
incon_tranfusion_ffd3 <- incon(tranfusion_ffd3)

tranfusion_ffd4 <- read_csv('tranfusion_ffd4.csv') # 
incon_tranfusion_ffd4 <- incon(tranfusion_ffd4)