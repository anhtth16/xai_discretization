# Calculate inconsistency rate after discretization
# Project: Capita Selecta
# Datasets: pageblock
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

#--- CHECK: pageblock
## EWD
pageblock_ewd1 <- read_csv('pageblock_ewd1.csv')
incon_pageblock_ewd1 <- incon(pageblock_ewd1) # 

pageblock_ewd2 <- read_csv('pageblock_ewd2.csv')
incon_pageblock_ewd2 <- incon(pageblock_ewd2) # 

pageblock_ewd3 <- read_csv('pageblock_ewd3.csv')
incon_pageblock_ewd3 <- incon(pageblock_ewd3) # 

## EFD
pageblock_efd1 <- read_csv('pageblock_efd1.csv') #
incon_pageblock_efd1 <- incon(pageblock_efd1)

pageblock_efd2 <- read_csv('pageblock_efd2.csv') # 
incon_pageblock_efd2 <- incon(pageblock_efd2)

pageblock_efd3 <- read_csv('pageblock_efd3.csv') # 
incon_pageblock_efd3 <- incon(pageblock_efd3)

# FFD
pageblock_ffd1 <- read_csv('pageblock_ffd1.csv') # 
incon_pageblock_ffd1 <- incon(pageblock_ffd1)

pageblock_ffd2 <- read_csv('pageblock_ffd2.csv') # 
incon_pageblock_ffd2 <- incon(pageblock_ffd2)

pageblock_ffd3 <- read_csv('pageblock_ffd3.csv') # 
incon_pageblock_ffd3 <- incon(pageblock_ffd3)

pageblock_ffd4 <- read_csv('pageblock_ffd4.csv') # 
incon_pageblock_ffd4 <- incon(pageblock_ffd4)