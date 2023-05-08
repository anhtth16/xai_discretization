# Calculate inconsistency rate after discretization
# Project: Capita Selecta
# Datasets: pageblock
# Discretization: Supervised (ChiMerge, DecisionTree)
# Source: R documentation, library chiM
# Update: 23.03.23, by Sam

# Load library
library(readr)
library(discretization)

# # Sample code
# #--Discretization using the ChiMerge method
# data(pageblock)
# disc=chiM(pageblock,alpha=0.05)
# #--cut-points
# disc$cutp

#--CALCULATING INCONSISTENCY RATE
# The inconsistency rate of dataset is calculated as follows:
# (1) two instances are considered inconsistent if they match except for their class labels;
# (2) for all the matching instances (without considering their class labels),
# the inconsistency count is the number of the instances minus the largest number of instances of class labels;
# (3) the inconsistency rate is the sum of all the inconsistency counts
# divided by the total number of instances.

# dispageblock=disc$Disc.data
# icon_pageblock_sample <- incon(dispageblock) #

#--- CHECK pageblock datasets
## ChiMerge
pageblock_cm1 <- read_csv('sc_cm_pageblock_6int.csv')
incon_pageblock_cm1 <- incon(pageblock_cm1) # 0.009683903

pageblock_cm2 <- read_csv('sc_cm_pageblock_8int.csv')
incon_pageblock_cm2 <- incon(pageblock_cm2) # 0.00566417

pageblock_cm3 <- read_csv('sc_cm_pageblock_10int.csv')
incon_pageblock_cm3 <- incon(pageblock_cm3) # 0.003837018

pageblock_cm4 <- read_csv('sc_cm_pageblock_15int.csv')
incon_pageblock_cm4 <- incon(pageblock_cm4) # 0.003654303

## DecisionTree
pageblock_dt1 <- read_csv('DT_small_discretized_pageblock.csv')
incon_pageblock_dt1 <- incon(pageblock_dt1) # 0.02302211

pageblock_dt2 <- read_csv('DT_medium_discretized_pageblock.csv')
incon_pageblock_dt2 <- incon(pageblock_dt2) # 0.01023205

pageblock_dt3 <- read_csv('DT_large_discretized_pageblock.csv')
incon_pageblock_dt3 <- incon(pageblock_dt3) # 0.004750594

pageblock_dt4 <- read_csv('DT_verylarge_discretized_pageblock.csv')
incon_pageblock_dt4 <- incon(pageblock_dt4) # 0.003288873
