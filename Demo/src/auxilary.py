import pandas as pd
import os

def create_demo(fileName='csv_example.csv'):
    '''
        Creates a csv file with the 7 points and the firs row includes the header
    '''
    my_dict = { 'image_name'    :[],
                #7 major points
                'Eye_br_L'      :[],
                'Eye_br_R'      :[],
                'Eye_soc_L'     :[],
                'Eye_soc_R'     :[],
                'Nostril_L'     :[],
                'Nostril_R'     :[],
                'Moustache'     :[],
                # 21 indeces
                'L_EB_to_R_EB'  :[],
                'L_EB_to_L_ES'  :[],
                'L_EB_to_R_ES'  :[],
                'L_EB_to_L_NT'  :[],
                'L_EB_to_R_NT'  :[],
                'L_EB_to_MCH'   :[],

                'R_EB_to_L_ES'  :[],
                'R_EB_to_R_ES'  :[],
                'R_EB_to_L_NT'  :[],
                'R_EB_to_R_NT'  :[],
                'R_EB_to_MCH'   :[],

                'L_ES_to_R_ES'  :[],
                'L_ES_to_L_NT'  :[],
                'L_ES_to_R_NT'  :[],
                'L_ES_to_R_MCH' :[],

                'R_ES_to_L_NT'  :[],
                'R_ES_to_R_NT'  :[],
                'R_ES_to_MCH'   :[],
                
                'L_NT_to_R_NT'  :[],
                'L_NT_to_MCH'   :[],

                'R_NT_to_MCH'   :[]
               
                }

    #convert it into dataframe
    df = pd.DataFrame(my_dict)
    #transfer into csv file
    df.to_csv(fileName, index=False)

def read_csv(fileName='csv_example.csv'):
    '''
        returns the csv file we're working on
    '''
    df = pd.read_csv('csv_example.csv')
    return df


def store_csv(dataframe,fileName='csv_example.csv'):
    '''
        this function is called implicity to store the new dictionary rows added.
    '''
    dataframe.to_csv(fileName, index=False)

def add_row (dataframe,row_dict ,fileName='csv_example.csv'):
    '''
        Append row of data, and store it.
    '''
    if len(row_dict) < 29:
        print ("error: row length is incorrect!", row_dict)
        return None

    row = pd.DataFrame(row_dict)
    dataframe_concatenatd = pd.concat([dataframe,row], ignore_index = True)
    store_csv(dataframe=dataframe_concatenatd, fileName=fileName)
    return dataframe_concatenatd


