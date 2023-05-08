# Calculate inconsistency rate after discretization
# Project: Capita Selecta
# Datasets: adult
# Discretization: Supervised (ChiMerge, DecisionTree)
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

# disadult=disc$Disc.data
# icon_adult_sample <- incon(disadult) #

#--- CHECK adult datasets
## ChiMerge
adult_cm1 <- read_csv('sc_cm_adult_6int.csv')
incon_adult_cm1 <- incon(adult_cm1) # 0.1579583

adult_cm2 <- read_csv('sc_cm_adult_8int.csv')
incon_adult_cm2 <- incon(adult_cm2) # 0.1498915

adult_cm3 <- read_csv('sc_cm_adult_10int.csv')
incon_adult_cm3 <- incon(adult_cm3) # 0.1401048

adult_cm4 <- read_csv('sc_cm_adult_15int.csv')
incon_adult_cm4 <- incon(adult_cm4) # 0.129663

## DecisionTree
adult_dt1 <- read_csv('DT_small_discretized_adult.csv')
incon_adult_dt1 <- incon(adult_dt1) # 0.07305188

adult_dt2 <- read_csv('DT_medium_discretized_adult.csv')
incon_adult_dt2 <- incon(adult_dt2) # 0.05986651

adult_dt3 <- read_csv('DT_large_discretized_adult.csv')
incon_adult_dt3 <- incon(adult_dt3) # 0.04578027

adult_dt4 <- read_csv('DT_verylarge_discretized_adult.csv')
incon_adult_dt4 <- incon(adult_dt4) # 0.03144834
