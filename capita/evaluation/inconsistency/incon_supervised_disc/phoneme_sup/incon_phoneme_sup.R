# Calculate inconsistency rate after discretization
# Project: Capita Selecta
# Datasets: phoneme
# Discretization: Supervised (ChiMerge, DecisionTree)
# Source: R documentation, library chiM
# Update: 23.03.23, by Sam

# Load library
library(readr)
library(discretization)

# # Sample code
# #--Discretization using the ChiMerge method
# data(phoneme)
# disc=chiM(phoneme,alpha=0.05)
# #--cut-points
# disc$cutp

#--CALCULATING INCONSISTENCY RATE
# The inconsistency rate of dataset is calculated as follows:
# (1) two instances are considered inconsistent if they match except for their class labels;
# (2) for all the matching instances (without considering their class labels),
# the inconsistency count is the number of the instances minus the largest number of instances of class labels;
# (3) the inconsistency rate is the sum of all the inconsistency counts
# divided by the total number of instances.

# disphoneme=disc$Disc.data
# icon_phoneme_sample <- incon(disphoneme) #

#--- CHECK phoneme datasets
## ChiMerge
phoneme_cm1 <- read_csv('sc_cm_phoneme_6int.csv')
incon_phoneme_cm1 <- incon(phoneme_cm1) # 0.1432272

phoneme_cm2 <- read_csv('sc_cm_phoneme_8int.csv')
incon_phoneme_cm2 <- incon(phoneme_cm2) # 0.09011843

phoneme_cm3 <- read_csv('sc_cm_phoneme_10int.csv')
incon_phoneme_cm3 <- incon(phoneme_cm3) # 0.05866025

phoneme_cm4 <- read_csv('sc_cm_phoneme_15int.csv')
incon_phoneme_cm4 <- incon(phoneme_cm4) # 0.03164323

## DecisionTree
phoneme_dt1 <- read_csv('DT_small_discretized_phoneme.csv')
incon_phoneme_dt1 <- incon(phoneme_dt1) # 0.1626573

phoneme_dt2 <- read_csv('DT_medium_discretized_phoneme.csv')
incon_phoneme_dt2 <- incon(phoneme_dt2) # 0.1082531

phoneme_dt3 <- read_csv('DT_large_discretized_phoneme.csv')
incon_phoneme_dt3 <- incon(phoneme_dt3) # 0.05829016

phoneme_dt4 <- read_csv('DT_verylarge_discretized_phoneme.csv')
incon_phoneme_dt4 <- incon(phoneme_dt4) # 0.0388601
