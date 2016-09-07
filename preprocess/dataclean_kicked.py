import pandas as pd
def preprocess(dataframe):
    df = dataframe
    df = df.drop(['RefId'],axis =1)
    #print df
    df['AuctionAve'] = sum(df[ave] for ave in auction_averages) /len(auction_averages)
    df = df.drop(auction_averages, axis=1)
    #print df
    for col in num_cols:
        df[col]=df[col].fillna(df[col].median())
    for col in nominal_cols:
        mode = df[col].mode()[0]
        df[col]=df[col].fillna(mode)
    print df
    df = df.drop(['PurchDate'], axis=1)
    df = df.drop(['VehYear'], axis=1)
    df = df.drop(['VNZIP1'],axis =1)
    df = df.drop(['AUCGUART'],axis =1)
    df = df.drop(['PRIMEUNIT'],axis =1)
    df.to_csv('training_pre.csv', sep ='\t')

train_dataframe = pd.read_csv('training.csv',header =0)

auction_averages = ['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
                        'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
                        'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
                        'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice']

nominal_cols =['Auction', 'Make', 'Trim', 'TopThreeAmericanName', 'Model', 'SubModel', 'Color', 'Transmission', 'WheelType',
                'PRIMEUNIT', 'AUCGUART', 'Nationality', 'Size', 'VNST']

num_cols = ['VehicleAge', 'WheelTypeID', 'VehYear','AuctionAve',
            'VehBCost','VehOdo', 'BYRNO', 'VNZIP1', 'IsOnlineSale', 'WarrantyCost']

global df

preprocess(train_dataframe)