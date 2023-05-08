# Calculate inconsistency rate after discretization
# Project: Capita Selecta
# Datasets: tranfusion
# Discretization: Supervised (ChiMerge, DecisionTree)
# Source: R documentation, library chiM
# Update: 23.03.23, by Sam

# Load library
library(readr)
library(discretization)

# # Sample code
# #--Discretization using the ChiMerge method
# data(tranfusion)
# disc=chiM(tranfusion,alpha=0.05)
# #--cut-points
# disc$cutp

#--CALCULATING INCONSISTENCY RATE
# The inconsistency rate of dataset is calculated as follows:
# (1) two instances are considered inconsistent if they match except for their class labels;
# (2) for all the matching instances (without considering their class labels),
# the inconsistency count is the number of the instances minus the largest number of instances of class labels;
# (3) the inconsistency rate is the sum of all the inconsistency counts
# divided by the total number of instances.

# distranfusion=disc$Disc.data
# icon_tranfusion_sample <- incon(distranfusion) #

#--- CHECK tranfusion datasets
## ChiMerge
tranfusion_cm1 <- read_csv('sc_cm_tranfusion_6int.csv')
incon_tranfusion_cm1 <- incon(tranfusion_cm1) # 0.1911765

tranfusion_cm2 <- read_csv('sc_cm_tranfusion_8int.csv')
incon_tranfusion_cm2 <- incon(tranfusion_cm2) # 0.1911765

tranfusion_cm3 <- read_csv('sc_cm_tranfusion_10int.csv')
incon_tranfusion_cm3 <- incon(tranfusion_cm3) # 0.1911765

tranfusion_cm4 <- read_csv('sc_cm_tranfusion_15int.csv')
incon_tranfusion_cm4 <- incon(tranfusion_cm4) # 0.1911765

## DecisionTree
tranfusion_dt1 <- read_csv('DT_small_discretized_tranfusion.csv')
incon_tranfusion_dt1 <- incon(tranfusion_dt1) # 0.2032086

tranfusion_dt2 <- read_csv('DT_medium_discretized_tranfusion.csv')
incon_tranfusion_dt2 <- incon(tranfusion_dt2) # 0.1951872

tranfusion_dt3 <- read_csv('DT_large_discretized_tranfusion.csv')
incon_tranfusion_dt3 <- incon(tranfusion_dt3) # 0.1778075

tranfusion_dt4 <- read_csv('DT_verylarge_discretized_tranfusion.csv')
incon_tranfusion_dt4 <- incon(tranfusion_dt4) # 0.157754
