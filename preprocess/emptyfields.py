import pandas as pd
def preprocess(dataframe):
    df = dataframe
    df = df.drop(['RefId'],axis =1)
    for col in num_cols:
        df[col]=df[col].fillna(df[col].median())
    for col in nominal_cols:
        mode = df[col].mode()[0]
        df[col]=df[col].fillna(mode)
    print df

    df.to_csv('training_pre', sep =',')

train_dataframe = pd.read_csv('training.csv',header =0)


nominal_cols =['Auction', 'Make', 'Trim', 'TopThreeAmericanName', 'Model', 'SubModel', 'Color', 'Transmission', 'WheelType',
                'PRIMEUNIT', 'AUCGUART', 'Nationality', 'Size', 'VNST']

num_cols = ['VehicleAge', 'WheelTypeID', 'VehYear','MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice',
            'MMRAcquisitionRetailAveragePrice','MMRAcquisitonRetailCleanPrice','MMRCurrentAuctionAveragePrice',
            'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice','MMRCurrentAuctionAveragePrice',
            'MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice','MMRCurrentRetailCleanPrice',
            'VehBCost','VehOdo', 'BYRNO', 'VNZIP1', 'IsOnlineSale', 'WarrantyCost']

global df

preprocess(train_dataframe)