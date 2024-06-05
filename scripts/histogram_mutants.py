import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

## Validation cohort dataframe
validation_AON = np.array([2,3,2,1,1,1,2,2,3,3,1,2])
validation_AON2 = np.array([2,3,2,1,1,1,2,1,2,2,1,1])
brain_area = ["AON", "AON","AON","AON","AON","AON","AON","AON","AON","AON","AON","AON",
              "AON2","AON2","AON2","AON2","AON2","AON2","AON2","AON2","AON2","AON2","AON2","AON2"]
validation_tr = np.array([6, 9, 5, 8, 4, 5, 7, 4, 7.5, 1, 7.5, 7.5]) #tube rank
validation_cr = np.array([9, 8.5, 2, 6, 7, 7, 3.5, 2, 9.5, 7, 2, 2]) # chasing rank
validation_cin = np.array([8, 8, 8, 8,8,8,8.5, 7, 8, 7, 8, 6.5]) # chasing in
validation_cout = np.array([5, 7, 9, 8, 8, 8, 9, 9, 2.5, 9, 8.5, 8.5]) # chasing out
# val_dict = {'tr':validation_tr, 'cr': validation_cr, 'cin': validation_cin, 'cout':validation_cout, 'aon':validation_AON, 'aon2':validation_AON2}
val_dict = {'Tube rank':np.concatenate((validation_tr,validation_tr)), 'Chasing rank': np.concatenate((validation_cr,validation_cr)),
                                'cin': np.concatenate((validation_cin,validation_cin)), 'Chasing out':np.concatenate((validation_cout,validation_cout)),
                                'Gene deletion':np.concatenate((validation_AON,validation_AON2)), 'Brain area': brain_area}
validation_df = pd.DataFrame(data=val_dict)

## First cohort dataframe
first_brain_area = ["AON", "AON","AON","AON","AON","AON","AON",
              "AON2","AON2","AON2","AON2","AON2","AON2","AON2"]
first_AON = np.array([0,0,1,2,3,3,3])
first_AON2 = np.array([0,0,0,1,2,2,3])
first_time = np.array([3114, 1950, 2480, 2337, 3228, 5224, 2507]) # time in arena
first_cr = np.array([3, 4.5, 8.5, 9, 10, 7, 2.5]) # chasing rank
first_tr = np.array([5.5, 8, 6, 8, 6, 3.5, 9])
first_dict = {'Tube rank':np.concatenate((first_tr, first_tr)), 'Chasing rank': np.concatenate((first_cr, first_cr)),
                            'Time in arena':np.concatenate((first_time, first_time)), 'Gene deletion':np.concatenate((first_AON, first_AON2)), 'Brain area': first_brain_area}
first_df = pd.DataFrame(data=first_dict)

## Both cohort combined
both_brain_area = first_brain_area
both_brain_area.extend(brain_area) #extend is in-place operation
both_dict = {'Tube rank':np.concatenate((first_tr, first_tr, validation_tr, validation_tr)), 
            'Chasing rank': np.concatenate((first_cr, first_cr, validation_cr, validation_cr)),
            'Gene deletion':np.concatenate((first_AON, first_AON2, validation_AON, validation_AON2)), 
            'Brain area': both_brain_area}
both_df = pd.DataFrame(data=both_dict)

def mutants_tube_rank(yaxis = "Chasing rank", dataset = "both", hue = True):
    sns.set(font_scale = 1.5)
    if dataset == "both":
        if hue:
            sns.catplot(data=both_df, x="Gene deletion", y=yaxis, hue="Brain area", kind = 'bar')
        else:
            sns.catplot(data=both_df, x="Gene deletion", y=yaxis, kind="bar", color = "gray")
            plt.title("Average effect over both brain areas")
            plt.tight_layout()

    elif dataset == "validation":
        if hue:
            sns.catplot(data=validation_df, x="Gene deletion", y=yaxis, hue="Brain area", kind = 'bar')
        else:
            sns.catplot(data=validation_df, x="Gene deletion", y=yaxis, kind = 'bar', color = "gray")
            plt.title("Average over AON & AON2")
            plt.tight_layout()

    elif dataset == "first":
        if hue:
            sns.catplot(data=first_df, x="Gene deletion", y=yaxis, hue="Brain area", kind = 'bar')
        else:
            sns.catplot(data=first_df, x="Gene deletion", y=yaxis, kind = 'bar', color = 'gray')

y = "Chasing rank"
mutants_tube_rank(y, dataset = "both", hue = True)
mutants_tube_rank(y, dataset = "both", hue = False)


