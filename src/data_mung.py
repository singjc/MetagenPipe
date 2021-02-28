#!/bin/python
import pandas as pd 
import numpy as np 
class data_mung:
    def __init__(self, sample_meta_file):
        self.sample_meta_file = sample_meta_file

    def group_age( self ):
        """
            Consume SampleMetaData, and make a new category to group age.
        """
        if False:
            sample_meta_file="/media/justincsing/ExtraDrive1/Documents2/Roest_Lab/Github/microbiome_OJS/data/preprocessed/Gupta_2020_Precompiled_Cleaned/SampleMetaDataClean.csv"
        data = pd.read_csv( self.sample_meta_file )
        data["age_group_tmp"] = data['Age (Years)']
        ## Bin into groups
        ### Note there are four other types of values: nan, not recorded, Adult and Adult Twins
        ### Assign nans -1
        data["age_group_tmp"][ data["age_group_tmp"].isna() ] = -1
        ### Assign not recorded -2
        data["age_group_tmp"][ data["age_group_tmp"]=="not recorded" ] = -2
        ### Assign Adult -3
        data["age_group_tmp"][ data["age_group_tmp"]=="Adult" ] = -3
        ### Assign Adult Twins -4
        data["age_group_tmp"][ data["age_group_tmp"]=="Adult Twins" ] = -4
        ### Remove * attached to some ages
        data['age_group_tmp'] = data['age_group_tmp'].replace({'\*': ''}, regex=True)
        data['age_group_tmp'] = data['age_group_tmp'].astype(float)
        
        ### Do age group binning
        data["age_group"] = data['Age (Years)']
        data["age_group"][ (data["age_group_tmp"]>=0) & (data["age_group_tmp"]<10) ] = "0-10"
        data["age_group"][ (data["age_group_tmp"]>=10) & (data["age_group_tmp"]<20) ] = "10-20"
        data["age_group"][ (data["age_group_tmp"]>=20) & (data["age_group_tmp"]<30) ] = "20-30"
        data["age_group"][ (data["age_group_tmp"]>=30) & (data["age_group_tmp"]<40) ] = "30-40"
        data["age_group"][ (data["age_group_tmp"]>=40) & (data["age_group_tmp"]<50) ] = "40-50"
        data["age_group"][ (data["age_group_tmp"]>=50) & (data["age_group_tmp"]<60) ] = "50-60"
        data["age_group"][ (data["age_group_tmp"]>=60) & (data["age_group_tmp"]<70) ] = "60-70"
        data["age_group"][ (data["age_group_tmp"]>=70) & (data["age_group_tmp"]<80) ] = "70-80"
        data["age_group"][ (data["age_group_tmp"]>=80) & (data["age_group_tmp"]<90) ] = "80-90"
        data["age_group"][ (data["age_group_tmp"]>=90) & (data["age_group_tmp"]<100) ] = "90-100"
        data["age_group"][ (data["age_group_tmp"]>=100) ] = "100+"
        ### Add other types
        data["age_group"][ (data["age_group_tmp"]==-1) ] = "unknown"
        data["age_group"][ (data["age_group_tmp"]==-2) ] = "unknown"
        data["age_group"][ (data["age_group_tmp"]==-3) ] = "unknown_adult"
        data["age_group"][ (data["age_group_tmp"]==-4) ] = "unknown_adult_twins"

        ### Drop temporary column
        data.drop(["age_group_tmp"], axis=1, inplace=True)
        print("INFO: Saving Age Groups to file...")
        data.to_csv( self.sample_meta_file )

    def categorize_age( self ):
        """
            Consume SampleMetaData, and make a new category to group age.
        """
        if False:
            sample_meta_file="/media/justincsing/ExtraDrive1/Documents2/Roest_Lab/Github/microbiome_OJS/data/preprocessed/Gupta_2020_Precompiled_Cleaned/SampleMetaDataClean.csv"
        data = pd.read_csv( self.sample_meta_file )
        data["age_group_tmp"] = data['Age (Years)']
        ## Bin into groups
        ### Note there are four other types of values: nan, not recorded, Adult and Adult Twins
        ### Assign nans -1
        data["age_group_tmp"][ data["age_group_tmp"].isna() ] = -1
        ### Assign not recorded -2
        data["age_group_tmp"][ data["age_group_tmp"]=="not recorded" ] = -2
        ### Assign Adult -3
        data["age_group_tmp"][ data["age_group_tmp"]=="Adult" ] = -3
        ### Assign Adult Twins -4
        data["age_group_tmp"][ data["age_group_tmp"]=="Adult Twins" ] = -4
        ### Remove * attached to some ages
        data['age_group_tmp'] = data['age_group_tmp'].replace({'\*': ''}, regex=True)
        data['age_group_tmp'] = data['age_group_tmp'].astype(float)
        
        ### Do age group binning
        data["age_category"] = data['Age (Years)']
        data["age_category"][ (data["age_group_tmp"].round()>=0) & (data["age_group_tmp"].round()<=14) ] = "child"
        data["age_category"][ (data["age_group_tmp"].round()>=15) & (data["age_group_tmp"].round()<=24) ] = "youth"
        data["age_category"][ (data["age_group_tmp"].round()>=25) & (data["age_group_tmp"].round()<=64) ] = "adult"
        data["age_category"][ (data["age_group_tmp"].round()>=65) ] = "senior"
        ### Add other types
        data["age_category"][ (data["age_group_tmp"]==-1) ] = "unknown"
        data["age_category"][ (data["age_group_tmp"]==-2) ] = "unknown"
        data["age_category"][ (data["age_group_tmp"]==-3) ] = "adult"
        data["age_category"][ (data["age_group_tmp"]==-4) ] = "adult"

        ### Drop temporary column
        data.drop(["age_group_tmp"], axis=1, inplace=True)
        print("INFO: Saving Age Categories to file...")
        data.to_csv( self.sample_meta_file  )