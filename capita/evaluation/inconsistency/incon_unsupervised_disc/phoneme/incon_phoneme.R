# Calculate inconsistency rate after discretization
# Project: Capita Selecta
# Datasets: phoneme
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

#--- CHECK: phoneme
## EWD
phoneme_ewd1 <- read_csv('phoneme_ewd1.csv')
incon_phoneme_ewd1 <- incon(phoneme_ewd1) # 

phoneme_ewd2 <- read_csv('phoneme_ewd2.csv')
incon_phoneme_ewd2 <- incon(phoneme_ewd2) # 

phoneme_ewd3 <- read_csv('phoneme_ewd3.csv')
incon_phoneme_ewd3 <- incon(phoneme_ewd3) # 

## EFD
phoneme_efd1 <- read_csv('phoneme_efd1.csv') #
incon_phoneme_efd1 <- incon(phoneme_efd1)

phoneme_efd2 <- read_csv('phoneme_efd2.csv') # 
incon_phoneme_efd2 <- incon(phoneme_efd2)

phoneme_efd3 <- read_csv('phoneme_efd3.csv') # 
incon_phoneme_efd3 <- incon(phoneme_efd3)

# FFD
phoneme_ffd1 <- read_csv('phoneme_ffd1.csv') # 
incon_phoneme_ffd1 <- incon(phoneme_ffd1)

phoneme_ffd2 <- read_csv('phoneme_ffd2.csv') # 
incon_phoneme_ffd2 <- incon(phoneme_ffd2)

phoneme_ffd3 <- read_csv('phoneme_ffd3.csv') # 
incon_phoneme_ffd3 <- incon(phoneme_ffd3)

phoneme_ffd4 <- read_csv('phoneme_ffd4.csv') # 
incon_phoneme_ffd4 <- incon(phoneme_ffd4)
