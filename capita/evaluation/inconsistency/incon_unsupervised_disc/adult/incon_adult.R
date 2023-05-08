# Calculate inconsistency rate after discretization
# Project: Capita Selecta
# Datasets: adult
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
## EWD
adult_ewd1 <- read_csv('adult_ewd1.csv')
incon_adult_ewd1 <- incon(adult_ewd1) # 

adult_ewd2 <- read_csv('adult_ewd2.csv')
incon_adult_ewd2 <- incon(adult_ewd2) # 

adult_ewd3 <- read_csv('adult_ewd3.csv')
incon_adult_ewd3 <- incon(adult_ewd3) # 

## EFD
adult_efd1 <- read_csv('adult_efd1.csv')
incon_adult_efd1 <- incon(adult_efd1) # 

adult_efd2 <- read_csv('adult_efd2.csv')
incon_adult_efd2 <- incon(adult_efd2) # 

adult_efd3 <- read_csv('adult_efd3.csv')
incon_adult_efd3 <- incon(adult_efd3) # 

## FFD
adult_ffd1 <- read_csv('adult_ffd1.csv')
incon_adult_ffd1 <- incon(adult_ffd1) # 

adult_ffd2 <- read_csv('adult_ffd2.csv')
incon_adult_ffd2 <- incon(adult_ffd2) # 

adult_ffd3 <- read_csv('adult_ffd3.csv')
incon_adult_ffd3 <- incon(adult_ffd3) # 

adult_ffd4 <- read_csv('adult_ffd4.csv')
incon_adult_ffd4 <- incon(adult_ffd4) # 

