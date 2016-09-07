library(ggplot2)
ca.train <-read.csv("/users/ramarvab/desktop/training.csv",header = TRUE)
summary(ca.train)
str(ca.train)

table(ca.train$IsBadBuy)
table(ca.train$Auction)

#Data Exploration
#Vehicle Age
ggplot(ca.train, aes(x = VehicleAge)) + geom_line(stat = "density") + ggtitle("Vehicle Age Density Curve")
ggplot(ca.train, aes(x = VehicleAge)) + geom_histogram() + ggtitle("Vehicle Age Histogram")
boxplot(ca.train$VehicleAge, main = "Vehicle Age Boxplot")

ggplot(ca.train, aes(x=VehOdo)) + geom_line(stat = "density") + ggtitle("Vehicle Odometer Density Curve")

ggplot(ca.train, aes(x=WarrantyCost)) + geom_line(stat = "density") + ggtitle("Warranty Cost Density Curve")

#Acquisition cost paid for the vehicle at time of purchase
ggplot(ca.train, aes(x=VehBCost)) + geom_line(stat = "density") + ggtitle("Acquisition Cost at Time of Purchase Cost Density Curve")


ggplot(ca.train, aes(x = Auction)) + geom_histogram() + ggtitle("Auction Histogram")

ggplot(ca.train, aes(x = Transmission)) + geom_histogram() + ggtitle("Transmission Histogram")

#recode the auction price variables from factor to numeric
ca.train$MMRAcquisitionAuctionAveragePrice<-as.numeric(ca.train$MMRAcquisitionAuctionAveragePrice)
#Acquisition price for this vehicle in average condition at time of purchase

ca.train$MMRAcquisitionAuctionCleanPrice <- as.numeric(ca.train$MMRAcquisitionAuctionCleanPrice)
#Acquisition price for this vehicle in the above Average condition at time of purchase

ca.train$MMRAcquisitionRetailAveragePrice <- as.numeric(ca.train$MMRAcquisitionRetailAveragePrice)
#Acquisition price for this vehicle in the retail market in average condition at time of purchase
ca.train$MMRAcquisitonRetailCleanPrice <- as.numeric(ca.train$MMRAcquisitonRetailCleanPrice)
#Acquisition price for this vehicle in the retail market in above average condition at time of purchase
ca.train$MMRCurrentAuctionAveragePrice <- as.numeric(ca.train$MMRCurrentAuctionAveragePrice)
#Acquisition price for this vehicle in average condition as of current day
ca.train$MMRCurrentAuctionCleanPrice <- as.numeric(ca.train$MMRCurrentAuctionCleanPrice)
#Acquisition price for this vehicle in the above condition as of current day
ca.train$MMRCurrentRetailAveragePrice <- as.numeric(ca.train$MMRCurrentRetailAveragePrice)
#Acquisition price for this vehicle in the retail market in average condition as of current day
ca.train$MMRCurrentRetailCleanPrice <- as.numeric(ca.train$MMRCurrentRetailCleanPrice)
#Acquisition price for this vehicle in the retail market in above average condition as of current day

ggplot(ca.train, aes(x=MMRAcquisitionAuctionAveragePrice)) + geom_line(stat = "density")
ggplot(ca.train, aes(x=MMRAcquisitionAuctionCleanPrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRAcquisitionRetailAveragePrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRAcquisitonRetailCleanPrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRCurrentAuctionAveragePrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRCurrentAuctionCleanPrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRCurrentRetailAveragePrice)) + geom_line(stat = "density") 
ggplot(ca.train, aes(x=MMRCurrentRetailCleanPrice)) + geom_line(stat = "density") 

#Cleaning the data set

#Redundant variable in Transmission
table(ca.train$Transmission)
ca.train$Transmission[ca.train$Transmission == "Manual"] <- "MANUAL"
table(ca.train$Transmission)