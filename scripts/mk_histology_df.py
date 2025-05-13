import pandas as pd
import numpy as np


## making histology df
def histology_info_df():
    first_cohort_path = "C:\\Users\\wolfgang.kelsch\\Documents\\GitHub\\RichClubs\\data\\reduced_data.xlsx"
    mutants1 = pd.read_excel(first_cohort_path)
    mutants1 = mutants1.drop_duplicates(subset=['Mouse_RFID'])[["Mouse_RFID", "mutant", "histology"]]
    
    
    second_cohort_path =  "C:\\Users\\wolfgang.kelsch\\Documents\\GitHub\\RichClubs\\data\\validation_cohort.xlsx"
    df2 = pd.read_excel(second_cohort_path)
    mutants2 = df2.rename(columns = {"genotype":"mutant"})
    mutants2.loc[mutants2['mutant'] == "Oxt", 'mutant'] = True
    mutants2.loc[mutants2['mutant'] == "WT", 'mutant'] = False
    mutants2 = mutants2.drop_duplicates(subset=['Mouse_RFID'])[["Mouse_RFID", "mutant", "histology"]]
    
    full_data = pd.concat((mutants1, mutants2))
    full_data.to_csv("C:\\Users\\wolfgang.kelsch\\Documents\\GitHub\\RichClubs\\data\\histology_classification.csv", 
                     index = False)

## making meta data df
def meta_data_df():
    first_cohort_path = "C:\\Users\\wolfgang.kelsch\\Documents\\GitHub\\RichClubs\\data\\reduced_data.xlsx"
    mutants1 = pd.read_excel(first_cohort_path)
    mutants1 = mutants1.rename(columns = {"group":"Group_ID"})
    mutants1 = mutants1[["Mouse_RFID", "Group_ID", "mutant", "histology", "RC"]]
    
    
    second_cohort_path =  "C:\\Users\\wolfgang.kelsch\\Documents\\GitHub\\RichClubs\\data\\validation_cohort.xlsx"
    df2 = pd.read_excel(second_cohort_path)
    mutants2 = df2.rename(columns = {"genotype":"mutant"})
    mutants2.loc[mutants2['mutant'] == "Oxt", 'mutant'] = True
    mutants2.loc[mutants2['mutant'] == "WT", 'mutant'] = False
    mutants2 = mutants2[["Mouse_RFID", "Group_ID", "mutant", "histology", "RC"]]
    
    full_data = pd.concat((mutants1, mutants2))
    full_data.to_csv("C:\\Users\\wolfgang.kelsch\\Documents\\GitHub\\RichClubs\\data\\meta_data.csv", 
                     index = False)
    
    
meta_data_df()