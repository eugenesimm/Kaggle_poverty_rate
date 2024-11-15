import pandas as pd
import os

# This script is used to perform initial cleaning of training datasets
RAW_DATA_DIR = "../data/raw/"
CLEAN_DATA_DIR = "../data/clean/"

def clean_Education(output_filename="train_Education_clean.csv"):
    '''
    Initially, train_Education.csv comes in the format:
        psu,hh,idcode,q01,q02, ...,Q65,Q66
    We want to clean the dataset to the following:
        psu_hh_idcode,Q01,Q02,...,Q65,Q66
    
    - combining psu,hh,idcode => psu_hh_idcode
    - standardizing explanatory column names to capital Qxx.
    '''

    # Read raw dataset
    Education = pd.read_csv(os.path.join(RAW_DATA_DIR, "train_Education.csv"))
    
    # psu,hh,idcode => psu_hh_idcode
    Education['psu_hh_idcode'] = Education['psu'].astype(str) + '_' + Education['hh'].astype(str) + '_' + Education['idcode'].astype(str)
    Education.drop(columns=['psu', 'hh', 'idcode'], inplace=True)

    # Reorder columns
    Education = Education[['psu_hh_idcode'] + [col for col in Education.columns if col != 'psu_hh_idcode']]
    
    # standardize column names qxx, Qxx => Qxx 
    Education.columns = [col.capitalize() if col.lower().startswith('q') else col for col in Education.columns]

    # IMPUTE MISSING VALUES HERE...
    

    # output dataframe to clean csv file
    Education.to_csv(os.path.join(CLEAN_DATA_DIR, output_filename))


def clean_HouseholdInfo(output_filename="train_HouseholdInfo_clean.csv"):
    '''
    clean_HousesholdInfo()
    '''
    HouseholdInfo = pd.read_csv(os.path.join(RAW_DATA_DIR, "train_HouseholdInfo.csv"))
    HouseholdInfo['psu_hh_idcode'] = HouseholdInfo['psu'].astype(str) + '_' + HouseholdInfo['hh'].astype(str) + '_' + HouseholdInfo['idcode'].astype(str)
    HouseholdInfo = HouseholdInfo[['psu_hh_idcode'] + [col for col in HouseholdInfo.columns if col != 'psu_hh_idcode']]
    HouseholdInfo.drop(columns=['psu', 'idcode', 'hh'], inplace=True)


    # output dataframe to clean csv file
    HouseholdInfo.to_csv(os.path.join(CLEAN_DATA_DIR, "train_HouseholdInfo_clean.csv"))


def clean_SubjectivePoverty(output_filename="train_SubjectivePoverty_clean.csv"):
    SubjectivePoverty = pd.read_csv(os.path.join(RAW_DATA_DIR, "train_SubjectivePoverty.csv"))
    subjective_poverty_columns = [f'subjective_poverty_{i}' for i in range(1, 11)]
    SubjectivePoverty['subjectivePoverty_rating'] = SubjectivePoverty[subjective_poverty_columns].idxmax(axis=1).str.extract('(\d+)').astype(int)
    Ratings = SubjectivePoverty[['psu_hh_idcode', 'subjectivePoverty_rating']]

    # output
    Ratings.to_csv(os.path.join(CLEAN_DATA_DIR, output_filename))

def merge_clean_datasets():
    pass




if __name__ == '__main__':
    clean_Education()
    clean_HouseholdInfo()
    clean_SubjectivePoverty()
    