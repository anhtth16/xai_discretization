# Calculate inconsistency rate after discretization
# Source: R documentation, library chiM
# Update: 28.06.22

# Load library
library(readr)
library(discretization)

# Sample code
#--Discretization using the ChiMerge method
data(iris)
disc=chiM(iris,alpha=0.05)
#--cut-points
disc$cutp

#--CALCULATING INCONSISTENCY RATE
# The inconsistency rate of dataset is calculated as follows:
# (1) two instances are considered inconsistent if they match except for their class labels;
# (2) for all the matching instances (without considering their class labels),
# the inconsistency count is the number of the instances minus the largest number of instances of class labels;
# (3) the inconsistency rate is the sum of all the inconsistency counts
# divided by the total number of instances.

disiris=disc$Disc.data
icon_iris_sample <- incon(disiris) # 0.02666667

#--- CHECK iris datasets
## EWD
iris_ewd1 <- read_csv('iris_ewd1.csv')
incon_iris_ewd1 <- incon(iris_ewd1) # 0.06666667

iris_ewd2 <- read_csv('iris_ewd2.csv')
incon_iris_ewd2 <- incon(iris_ewd2) # 0.02

iris_ewd3 <- read_csv('iris_ewd3.csv')
incon_iris_ewd3 <- incon(iris_ewd3) # 0.006666667

## EFD
iris_efd1 <- read_csv('iris_efd1.csv')
incon_iris_efd1 <- incon(iris_efd1) # 0.04

iris_efd2 <- read_csv('iris_efd2.csv')
incon_iris_efd2 <- incon(iris_efd2) # 0.04

iris_efd3 <- read_csv('iris_efd3.csv')
incon_iris_efd3 <- incon(iris_efd3) # 0

## FFD
iris_ffd1 <- read_csv('iris_ffd1.csv')
incon_iris_ffd1 <- incon(iris_ffd1) # 0

iris_ffd2 <- read_csv('iris_ffd2.csv')
incon_iris_ffd2 <- incon(iris_ffd2) # 0.006666667

iris_ffd3 <- read_csv('iris_ffd3.csv')
incon_iris_ffd3 <- incon(iris_ffd3) # 0.1866667

iris_ffd4 <- read_csv('iris_ffd4.csv')
incon_iris_ffd4 <- incon(iris_ffd4) # 0.1866667

#--- CHECK: pendigits
## EWD
pendigits_ewd1 <- read_csv('pendigits_ewd1.csv')
incon_pendigits_ewd1 <- incon(pendigits_ewd1) # 0.004275837

pendigits_ewd2 <- read_csv('pendigits_ewd2.csv')
incon_pendigits_ewd2 <- incon(pendigits_ewd2) # 9.097525e-05

pendigits_ewd3 <- read_csv('pendigits_ewd3.csv')
incon_pendigits_ewd3 <- incon(pendigits_ewd3) # 0

## EFD
pendigits_efd1 <- read_csv('pendigits_efd1.csv') # 0.001091703
incon_pendigits_efd1 <- incon(pendigits_efd1)

pendigits_efd2 <- read_csv('pendigits_efd2.csv')
incon_pendigits_efd2 <- incon(pendigits_efd2) # 0

pendigits_efd3 <- read_csv('pendigits_efd3.csv')
incon_pendigits_efd3 <- incon(pendigits_efd3) # 0

## FFD
pendigits_ffd1 <- read_csv('pendigits_ffd1.csv') # 0
incon_pendigits_ffd1 <- incon(pendigits_ffd1)

pendigits_ffd2 <- read_csv('pendigits_ffd2.csv') # 0
incon_pendigits_ffd2 <- incon(pendigits_ffd2)

pendigits_ffd3 <- read_csv('pendigits_ffd3.csv')
incon_pendigits_ffd3 <- incon(pendigits_ffd3) # 0

pendigits_ffd4 <- read_csv('pendigits_ffd4.csv')
incon_pendigits_ffd4 <- incon(pendigits_ffd4) # 0

#--- CHECK: satimage
## EWD
satimage_ewd1 <- read_csv('satimage_ewd1.csv')
incon_satimage_ewd1 <- incon(satimage_ewd1) # 0.01600622

satimage_ewd2 <- read_csv('satimage_ewd2.csv')
incon_satimage_ewd2 <- incon(satimage_ewd2) # 0.002175602

satimage_ewd3 <- read_csv('satimage_ewd3.csv')
incon_satimage_ewd3 <- incon(satimage_ewd3) # 0

## EFD
satimage_efd1 <- read_csv('satimage_efd1.csv') # 0.0007770008
incon_satimage_efd1 <- incon(satimage_efd1)

satimage_efd2 <- read_csv('satimage_efd2.csv') # 0
incon_satimage_efd2 <- incon(satimage_efd2)

satimage_efd3 <- read_csv('satimage_efd3.csv') # 0
incon_satimage_efd3 <- incon(satimage_efd3)

# FFD
satimage_ffd1 <- read_csv('satimage_ffd1.csv') # 0
incon_satimage_ffd1 <- incon(satimage_ffd1)

satimage_ffd2 <- read_csv('satimage_ffd2.csv') # 0
incon_satimage_ffd2 <- incon(satimage_ffd2)

satimage_ffd3 <- read_csv('satimage_ffd3.csv') # 0
incon_satimage_ffd3 <- incon(satimage_ffd3)

satimage_ffd4 <- read_csv('satimage_ffd4.csv') # 0
incon_satimage_ffd4 <- incon(satimage_ffd4)
