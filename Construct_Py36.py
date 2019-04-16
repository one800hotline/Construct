{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 0
}

    
import datetime as dt
import os as os

def f_welcome(exec_f
              ,platform):
    
    if exec_f:
        import os as os
        import datetime as dt
        import getpass

        print ("------------------------------------------")
        print ("Loading %s utility platform!" % platform)
        print ("------------------------------------------")
        
        # Set some parameters
        date = dt.datetime.today().strftime("%y-%m-%d")
        time = dt.datetime.today().strftime("%H:%M:%S")
        time_h = dt.datetime.today().hour
        
        # Are we fetching user (Linux) or username (Windows)?
        try:
            user = getpass.getuser()
        except:
            pass

        try:
            user = getpass.getuser()
        except:
            pass
        
        # Some nice welcoming rules based on time of day
        if 6 < time_h <= 12:
            
            print ("Good morning, %s, the time is %s and the date is: %s" % (user, time, date))
            
        elif 12 < time_h <= 18:
            
            print ("Good afternoon, %s, the time is %s and the date is: %s" % (user, time, date))
        
        elif 18 < time_h <= 24:
            
            print ("Good evening, %s, the time is %s and the date is: %s" % (user, time, date))
            
        else:
            
            print ("Its late, go home... ")
        
    else:
        print ("No execution, ending...")
        
        
f_welcome(True
         ,'Construct')
    
    


#--------------------------------- 
# TIME DIMENSIONS
#---------------------------------
from datetime import datetime, timedelta, date
import dateutil.relativedelta as rtd

def f_this_ym():
    return dt.date.today().strftime('%Y%m')

def f_last_ym():
    return (dt.date.today() - rtd.relativedelta(months=+1)).strftime("%Y%m")

def f_last_ymd_01():
    return (dt.date.today() - rtd.relativedelta(months=+1)  - rtd.relativedelta(day = 1)).strftime("%Y%m%d")
    
def f_t_now():
    return dt.datetime.today().strftime("%H:%M:%S")

def f_d_now():
    return dt.date.today().strftime("%Y-%m-%d")

def f_dt_now():
    return dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")


#-------------------------
# Construct functions
#-------------------------


# Define function for estimating WoE for a discrete variable
def f_woe(exec_f, indata, list_col_woe, dict_woe_summary):
    
    """
    This function maps WoE values to a discrete ordinal or nominal categorical variable. The function also pushes back a "long form" table containing all categorical features, their
    respective values, adherent WoE values and IV values. Finally, the function return an aggregated IV table where one can do EDA to review predictive strength of the inididual 
    feature with respect to target variable.
    
    """

    if exec_f:
    
        df_tmp=indata.copy()
        
        
        
        #-----------------------------------------------
        # One input feature to calculate WoE and IV for
        #-----------------------------------------------
        if len(list_col_woe)==1:
        
            # Aggregate discrete variable and calculate metrics
            df_woe=cstr.f_grpby_aug(True, df_tmp, list_col_woe, dict_woe_summary).rename(columns={'count':'N'
                                                                                                  ,'target_sum':'target_1'
                                                                                                 ,list_col_woe[0]:'var_lvls'})
            
            df_woe['var_name']=list_col_woe[0]

        #-----------------------------------------------
        # Mor ethan one feature, we create consolidated 
        # aggregation tables for WoE and IV 
        #-----------------------------------------------
        else:
                    
            list_tmp_hld_woe=list()
        
            # Itterate over features requiring WoE and IV calculation
            for idx_col_woe, col_woe in enumerate(list_col_woe):
                
                print ("Commencing calculation of WoE and IV for variable: {}".format(col_woe))
               
                # Aggregate discrete variable and calculate metrics
                df_tmp_woe=cstr.f_grpby_aug(True, df_tmp, [col_woe], dict_woe_summary).rename(columns={'count':'N'
                                                                                            ,'target_sum':'target_1'
                                                                                            ,col_woe:'var_lvls'})                
                
                df_tmp_woe['var_name']=col_woe
                
                # Common holder
                list_tmp_hld_woe.append(df_tmp_woe)
                
            # Concat to common woe
            df_woe=pd.concat(list_tmp_hld_woe, axis=0)

        #--------------------------------
        # Generic row level calculations
        #--------------------------------
        df_woe['target_0']=df_woe['N']-df_woe['target_1']

        df_woe['p_1']=df_woe['target_1'].div(df_woe['target_1'].sum(axis=0))
        df_woe['p_0']=df_woe['target_0'].div(df_woe['target_0'].sum(axis=0))

        # WoE and IV
        df_woe['woe']=np.log10(df_woe['p_1']/df_woe['p_0'])
        df_woe['iv']=(df_woe['p_1']-df_woe['p_0'])*df_woe['woe']


        # Sort order
        df_woe=df_woe[['var_name'] + [col for col in df_woe.columns if col not in (['var_name'])]]

        # calculate IV feature level
        df_iv=df_woe[['var_name','iv']].groupby(['var_name']).sum()
        

        return df_woe, df_iv
            
            
    else:
        print ("No execution of WoE function, passing indata.")
        return indata

    
def f_dict_map_woe(exec_f, indata_woe, list_col_map_woe):
    
    """
    This function creates a dictionary of dictionaries. The dictionary keys is name of discrete ordinal and nominal categorical features. Each of the features 
    is attached to another dictionary which holds each discrete level of the variable, and an adherent WoE mapping to be applied on a pandas DataFrame. The mapping
    is done through an itteration and map on the given feature.
    
    """

    if exec_f:
        
        df_tmp=indata_woe.copy()
    
        # Hold all dictionaries for all features
        dict_hld_woe=dict()

        # Itterate over columns in WoE table
        for idx_col_dict_woe, col_dict_woe in enumerate(list_col_map_woe):
            print ("Creating a dictionary mapping of input feature WoE values for {}".format(col_dict_woe))

            df_tmp_var_woe=df_tmp[df_tmp['var_name']==col_dict_woe]
            
            dict_tmp_woe=dict(zip(df_tmp_var_woe['var_lvls'].tolist()
                                    ,df_tmp_var_woe['woe'].tolist()))

            dict_hld_woe.update({col_dict_woe:dict_tmp_woe})
            
        return dict_hld_woe
            
    else:
        print ("No execution of function for mapping discrete featrue values to WoE values, ending...")




def f_desc_aug(exec_f, indata, list_col_trg, list_col_metric, n_bins):
    
    """
    Docstring:

    This function takes two input parameters, an array with continious elements and a binary target variabe. 
    The function return 3 object, a pd.describe() on vector, a pandas groupby table aggreation using decile bins,
    and a seaborn distribution plot using target as hue

    Parameters            
    exec_f                boolean(True/False): Execute function, or not
    df_temp               pd.DataFrame(), shape(n*m): indata containing features and target for analysis
    list_col_trg          list of string, 1 element: list with name of target column
    list_col_metric       list of string, 1 element: list with name of analysis metric
    n_bins             scalar, list or None: If scalar then equally sized bins. If list, the bin accordingly, if None then discrete levels of list_col_metric is used. 

    """
    
    if exec_f:
        import pandas as pd
        from IPython.core.display import display
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Set temporary working DataFrame
        df_temp=indata.copy()
        
        #----------------------------
        # Descriptive statistics
        #----------------------------
        print ("Describe:")
        display(pd.DataFrame(df_temp[list_col_metric[0]].describe()))


        #----------------------------
        # Bin and aggregate 
        #----------------------------

        # We have scalar instance of n_bins --> Create equal sized bins
        if isinstance(n_bins, int):
            # Binn
            df_temp['grp_'+list_col_metric[0]]=pd.qcut(df_temp[list_col_metric[0]], n_bins, duplicates='raise')
            df_temp['grp_'+list_col_metric[0]]=df_temp['grp_'+list_col_metric[0]].cat.codes

            # Aggrgate
            df_temp_agg=df_temp[list_col_trg+['grp_'+list_col_metric[0]]].groupby(['grp_'+list_col_metric[0]], as_index=False).agg(['count', 'sum', 'mean'])

        # We have a list instance of n_bins --> Create bins given list
        elif isinstance(n_bins, list):
            # Binn
            df_temp['grp_'+list_col_metric[0]]=pd.cut(df_temp[list_col_metric[0]], n_bins, include_lowest=True)
            df_temp['grp_'+list_col_metric[0]]=df_temp['grp_'+list_col_metric[0]].cat.codes

            # Aggrgate
            df_temp_agg=df_temp[list_col_trg+['grp_'+list_col_metric[0]]].groupby(['grp_'+list_col_metric[0]], as_index=False).agg(['count', 'sum', 'mean'])

        # n_bins is None, we use the discrete values of list_col_metric as aggregation
        else:
            print ("Ingoing metric is discrete, using its own levels for aggregation.")

            # Aggrgate
            df_temp_agg=df_temp[list_col_trg+[list_col_metric[0]]].groupby([list_col_metric[0]], as_index=False).agg(['count', 'sum', 'mean'])


        # Display aggregation
        display(df_temp_agg)


        #---------------------------------------------
        # Distribution, split on target
        #---------------------------------------------
        fig, ax=plt.subplots(ncols=1
                            ,nrows=1
                            ,figsize=(12,6))

        for trf_val in (df_temp[list_col_trg[0]].drop_duplicates()):
            sns.distplot(tuple(df_temp[df_temp[list_col_trg[0]]==trf_val][list_col_metric[0]])) 

        plt.tight_layout()
        plt.show()

        #------------------------
        # Scatter plot
        #------------------------
        
        # Using the ingoing metrics discrete levels as aggregation
        if n_bins==None:

            fig, ax=plt.subplots(ncols=1, nrows=1, figsize=(12, 6))
            sns.regplot(x=df_temp_agg.index.values, y=df_temp_agg[('target', 'mean')], order=3,ci=None)

            plt.tight_layout()
            plt.show()

        else:
            df_temp_agg=pd.concat([df_temp_agg, df_temp[['grp_'+list_col_metric[0], list_col_metric[0]]].groupby(['grp_'+list_col_metric[0]], as_index=False).mean()[list_col_metric[0]]]
                                    ,axis=1, sort=False)

            fig, ax=plt.subplots(ncols=1, nrows=1, figsize=(12, 6))
            sns.regplot(x=df_temp_agg.index.values, y=df_temp_agg[('target', 'mean')], order=3, ci=None)

            plt.tight_layout()
            plt.show()
            
    else:
    
        print ("No execution of function, ending....")

def f_regex_int(exec_f, series):
    """
    Takes object series, pushes back integers, [0-9] 
    """
    import pandas as pd
    return series.astype('str').str.extract(('(\d+)'), expand = True).astype('float64')


def f_regex_str(exec_f, series):
    """
    Takes object series, pushes back characters, [A-Ö, a-ä] 
    """
    import pandas as pd
    return series.astype('str').str.extract(('(\D+)'), expand = True)

def f_grpby_fillna(exec_f, indata, list_grpby, col_fillna, col_fillna_val, func_fillna):

    """

    Docstring for function used to fill missing values in discrete groups using groupby method and function accepted by transform

   

    Parameters:

   

    exec_f          Boolean(True/False): Execute function or not, in not return indata

    indata          pandas DataFrame (shape: m*n): Pandas DataFrame containing Series with missing value, and Series used to calculate Series ised for filling missing values.

    list_groupby    List of strings (shape: 1*n): N column names in string values, used to groupby data to create Series for filling missing values

    col_fillna      String: Column name to fill missing values in

    col_fillna_va   String: Column name used to calculate metric, given function, to be used for filling missing values

    func_fillna     String: Function accepted by transform for calculating fillna Series ('mean', 'median', numpy.sum, numpy.exp, etc, etc...)

   

    """

 

    # Go!

    if exec_f:

        import pandas as pd
        import numpy as np

        # Safety copy

        df_temp=indata.copy()

       

        # Perform fillna, inplace, with Series created by groupby and transform-step

        df_temp[col_fillna].fillna(df_temp[list_grpby+[col_fillna_val]].groupby(list_grpby).transform(func_fillna)[col_fillna_val]

                                  ,inplace=True)

       

        # Push back fillna DataFrame

        return df_temp

       

    # NoGo!

    else:

        print ("No execution of groupby fillna-function, returning indata...")

        return indata

    
def f_grpby_aug(exec_f,indata ,list_col_agg, dict_f_agg):

    """
    Docstring for augmented groupby:
    
    Used to apply multiple functions on indata given groupby aggregation. The multilevel hierarchical columns are flattened and combined for clear
    naming of columns and consistent metadata.
    
    exec_f            Boolean(True/False): Go or no go on function execution
    indata            pandas DataFrame(): Indata to be aggregated
    list_col_agg      list of strings - (n*1): Contains variable names of columns to be used for aggregation
    dict_f_agg        dictionary of key:value strings: Contains mapping of functions being applied to variables in indata    
    """
    
    # Go!
    if exec_f:
        import pandas as pd
        import numpy as np

        print ("Exeucting groupby function....")
        print ("\nAggregator variables are: \n{}".format(list_col_agg))
        print ("\nVariables with functons are: \n{}".format(dict_f_agg))
        
        # Perform aggregation on data
        indata_agg=indata[list_col_agg+list(dict_f_agg.keys())].groupby(list_col_agg, as_index=False).agg(dict_f_agg)

        # Get level 1 column names and set to list everything not being count and aggregators
        list_col_lvl1=list(indata_agg.columns.get_level_values(1))
        list_col_lvl1_mtrc=[col for col in list_col_lvl1 if col not in (['', 'count'])]

        # Get level 0 column names and set to list everything not being count and aggregators
        list_col_lvl0=list(indata_agg.columns.get_level_values(0))
        list_col_count=list({key:val for (key,val) in dict_f_agg.items() if val=='count'}.keys())

        list_col_lvl0_map=[col for col in list_col_lvl0 if col not in (list_col_agg+list_col_count)]
    
        # Set columns for aggregation DataFrame to level 1
        indata_agg.columns=indata_agg.columns.get_level_values(0)

        # Re-map everything using level 1 column names
        indata_agg.columns=list_col_agg+list_col_count + [col[0] + '_' + col[1] for col in zip(list_col_lvl0_map, list_col_lvl1_mtrc)]
        
        # Rename count column to 'count' and return
        indata_agg.rename(columns={key:val for (key,val) in dict_f_agg.items() if val=='count'}, inplace=True)
        
        return indata_agg
                
        
    # No go!
    else:
        print ("No execution of function, returining None.")
        return None
    
    
    
    
    
    
    
    
    
    
def f_cross_tab_aug(exec_f, indata,str_col_dim0, str_col_dim1, str_count_metric, bool_display=False):
    
    """
    Docstring:
    
    This functions does a count on elements within the matrix formed by row dimension and column dimension from str_col_dim0 and str_col_dim1 respectively.
    The function also pushes back a distribution of the grand total over the matrix as a second DataFrame
    
    exec_f            Boolean(True/False): Execute function, or not
    str_col_dim0      string: Dimension 0 aggregation (Index)
    str_col_dim1      string: Dimension 1 aggregation (Column)
    str_col_count     string: Dimension to count
    bool_display      boolean(True(False): If False, we return the two created DataFrames, if True we only display them - no return
    
    """

    if exec_f:
        import pandas as pd
        import numpy as np
        from IPython.core.display import display

        
        print ("Dim 0 aggregation: {}".format(str_col_dim0))
        print ("Dim 1 aggregation: {}".format(str_col_dim1))
        print ("Count dim is: {}".format(str_count_metric))

        df_temp=indata.copy()

        #---------------------------------
        # Main crosstab table
        #---------------------------------
        df_temp = pd.crosstab(df_temp[str_col_dim0], df_temp[str_col_dim1], values = str_count_metric, aggfunc='count').reset_index()

        # Create sum over axis 1
        df_temp['sum_axis_1'] = df_temp[[col for col in df_temp.columns[1:]]].sum(axis = 1)

        # Sort order
        df_temp2 = df_temp[[col for col in df_temp.columns if col not in (['sum_axis_1'])]]

        # Create one extra row to hold sum of axis 0
        df_temp2 = df_temp2[df_temp2.index == 0]

        # Set string aggregation column to blank
        df_temp2[[col for col in df_temp2.columns[0:1]]] = ''
        df_temp2.rename(columns={[col for col in df_temp2.columns[0:1]][0] : 'aggregation_dim'}, inplace = True)

        # All numeric column to nan
        df_temp2[[col for col in df_temp2.columns[1:]]] = np.nan

        # rename aggregation dim0 into 'aggregation dim'
        df_temp.rename(columns={[col for col in df_temp.columns[0:1]][0] : 'aggregation_dim'}, inplace = True)

        #---------------------------------
        # Common table for count
        #---------------------------------
        df_temp3 = pd.concat([df_temp, df_temp2]
                         ,axis = 0, sort=False
                         ).reset_index(drop = True)

        # Fill each numeric metric with its axis = 0 sum for each aggregation dim
        for idx_col_ax0, col_name_ax0 in enumerate([col for col in df_temp3.columns if col not in (['aggregation_dim', 'sum_axis_1'])]):
            print ("Sum for axis = 0, column: {}".format(col_name_ax0))
            df_temp3.loc[df_temp3['aggregation_dim'] == '', col_name_ax0] = df_temp3[col_name_ax0].sum(axis = 0)


        # Set sort order
        df_temp3 = df_temp3[['aggregation_dim'] + [col for col in df_temp3.columns if col not in (['aggregation_dim', 'sum_axis_1'])] + ['sum_axis_1']]

        # Set total sum for sum_axis_1
        df_temp3.loc[df_temp3['aggregation_dim'] == '', 'sum_axis_1'] = df_temp3['sum_axis_1'].sum(axis = 0)


        # Fill missing values
        df_temp3.fillna(0, inplace = True)


        #---------------------------------
        # Create % of DF
        #---------------------------------
        df_temp4 = df_temp3.copy()

        # Calculate % of total
        df_temp4[[col for col in df_temp4.columns if col != 'aggregation_dim']] = df_temp4[[col for col in df_temp4.columns if col != 'aggregation_dim']]/df_temp4[df_temp4['aggregation_dim'] == '']['sum_axis_1'].values[0]

        # return calculated dfs
        if bool_display==False:
            return df_temp3, df_temp4

        # Only display the output
        else:
            for idx_df, df_display in enumerate([df_temp3, df_temp4]):
                display(df_display)
            
        
    else:
        
        print ("No execution of funtion, ending....")


def f_wgt_avg(indata, wgt_col, avg_col, grp_col=None):

    """
    Docstring for weighted average calculation function.
    
    inparameters:
    
    df_temp         pd.DataFrame - (n*m): Pandas DataFrame with variables to make calculation on
    wgt_col         Array as string - (n*1): Variable used as weight in calculation of weighted average
    avg_col         Array as string - (n*1): Variable for calculating average value on
    grp_col         Array as string - (n*1)/None: Variable to used as discrete aggregation groups for doing weighted average calculation, if not None
    
    """
    import numpy as np
    import pandas as pd

    df_temp=indata.copy()
    
    # We calculate the weighted mean given discrete aggregations groups
    if grp_col is not None:

        # Weight variable multiplied by mean variable
        df_temp[wgt_col + '*' + avg_col] = df_temp[wgt_col] * df_temp[avg_col]

        # Sum of weight variable, by aggregation groups
        df_temp = df_temp[[grp_col,wgt_col, wgt_col + '*' + avg_col]].groupby(grp_col, as_index = False).sum()

        # Calculate weighted average, given groups
        df_temp['w_avg_' + avg_col] = df_temp[wgt_col + '*' + avg_col]/df_temp[wgt_col]

        # Push output
        print ("{} weights for avg. value calculation on variable {}, aggregation group-by column {}".format(wgt_col, avg_col, grp_col))
        return df_temp

    # Weighted mean is calculated on whole input data, given weight variable and mean variable
    elif grp_col is None:

        # Weight variable multiplied by mean variable
        df_temp[wgt_col + '*' + avg_col] = df_temp[wgt_col] * df_temp[avg_col]

        # Summarize
        df_temp = pd.DataFrame(df_temp[[wgt_col, avg_col, wgt_col + '*' + avg_col]].sum(axis = 0)).transpose()

        # Calculate mean
        df_temp['w_avg_' + wgt_col + '*' + avg_col] =  df_temp[wgt_col + '*' + avg_col]/df_temp[wgt_col]
        
        # Push output
        print ("{} weights for avg. value calculation on variable {}".format(wgt_col, avg_col))
        return df_temp    



def f_table_overv(exec_f, indata, list_col_excl):

    """
    Docstring:
    
    This function creates an overview of a DataFrame table, and calls pd.DataFrame().describe() on all numerical
    columns in the table. Output is pd.DataFrame of dim (n, m) where n is number of columns not being and excluded 
    by the list containing meta columns, and m is 14 (columns with descriptive information)
    
    exec_f              Boolean(True/False): Execute or not, if False then return None
    indata              pd.DataFrame: DataFrame to create overview of
    list_col_excl       list of dim (n, 1): List of columns that we dont want to include in overiview, e.g. id-kolumn, etc.


    """

    #----------------------------------
    # Data table overview 
    #----------------------------------

    # Go/No Go
    if exec_f:

        import pandas as pd
        import numpy as np
        
        
        # Column type into a series with index being column name
        srs_datatypes = indata[[col for col in indata.columns if col not in (list_col_excl)]].dtypes
        
        #-------------------------------------------
        # Numerical columns (Discrete + Continious)
        #-------------------------------------------

        # Boolean expression for taking out int + float rows from series
        boolean_srs_col_int = (srs_datatypes.astype(str).str.find('int') >= 0)
        boolean_srs_col_float = (srs_datatypes.astype(str).str.find('float') >= 0)

        # Put col name into list
        list_col_float = srs_datatypes[boolean_srs_col_float].index.tolist()
        list_col_int = srs_datatypes[boolean_srs_col_int].index.tolist()

        # All numeric columns
        list_col_num = list_col_float+list_col_int

        # Print out
        print ("Columns float: {}".format(list_col_float))
        print ("Columns int: {}".format(list_col_int))
        print ("\nAll numerical columns: {}".format(list_col_num))        
        
        
        # Holder for numeric columns
        list_hld_col_cnt_num = list()

        # Loop-through and categorize
        for idx_col_num, col_name_num in enumerate(list_col_num):

            # How many unique elements for a given column
            _col_elm_cnt = len(pd.unique(indata[col_name_num].values))

            # Append name + count to list holders
            list_hld_col_cnt_num.append((col_name_num, _col_elm_cnt))

        # List to DataFrame
        df_col_elem_cnt_num = pd.DataFrame(list_hld_col_cnt_num, columns = ['col_name', 'unq_val_cnt'])
        
        # Column names and datatypes
        df_col_dtype_num = pd.DataFrame(indata.dtypes).reset_index().rename(columns = {'index' : 'col_name', 0 : 'col_type'})
        df_col_dtype_num = df_col_dtype_num[df_col_dtype_num['col_name'].isin(list_col_num)].set_index('col_name')
        
        
        # Count all isnull
        srs_col_isnull = pd.DataFrame(indata[list_col_num].isnull().sum(axis = 0)).rename(columns = {0 : 'isnull_cnt'}).reset_index(drop = True)
        
        # Merge back count isnull + unique values + column data type, and have correct order on columns
        df_col_elem_cnt_num = pd.concat([df_col_elem_cnt_num[['col_name']], srs_col_isnull,df_col_elem_cnt_num[['unq_val_cnt']]]
                                   ,axis = 1, sort=False)

        # Count total nr rows
        df_col_elem_cnt_num['n_rows_tot'] = len(indata)
        
        # Calculate % missing values for given column
        df_col_elem_cnt_num['%_null_tot'] = df_col_elem_cnt_num['isnull_cnt'].div(df_col_elem_cnt_num['n_rows_tot'])

        # Copy table
        df_cols_meta_num = df_col_elem_cnt_num.copy()


        # Describe on all input columns 
        df_describe = indata[df_cols_meta_num.col_name.tolist()].describe().T

        # Set index to column col_name in meta table to enable join on index
        df_cols_meta_num.set_index('col_name', inplace=True)

        # Merge with metadata, exclude index and count
        df_cols_meta_num = pd.concat([df_cols_meta_num,df_col_dtype_num, df_describe[[col for col in df_describe.columns if col not in (['count'])]]]
                                ,axis = 1, sort=False).reset_index()

        # Final fix
        df_cols_meta_num.rename(columns={'index' : 'col_name'}, inplace = True)        
        df_cols_meta_num = df_cols_meta_num[['col_name', 'col_type'] + [col for col in df_cols_meta_num.columns if col not in (['col_name','col_type'])]]
        
        
        #------------------------------------
        # Other columns (String + date type)
        #------------------------------------

        boolean_srs_col_obj = (srs_datatypes.astype(str).str.find('obj') >= 0)
        boolean_srs_col_date = (srs_datatypes.astype(str).str.find('date') >= 0)

        list_col_obj = srs_datatypes[boolean_srs_col_obj].index.tolist()
        list_col_date = srs_datatypes[boolean_srs_col_date].index.tolist()

        list_col_oth = list_col_obj + list_col_date

        print ("Columns date: {}".format(list_col_date))
        print ("Columns object/string: {}".format(list_col_obj))
        print ("\nAll other columns: {}".format(list_col_oth))
        
        # Holder for numeric columns
        list_hld_col_cnt_oth = list()

        # Loop-through and categorize
        for idx_col_oth, col_name_oth in enumerate(list_col_oth):

            # How many unique elements for a given column
            _col_elm_cnt = len(pd.unique(indata[col_name_oth].values))

            # Append name + count to list holders
            list_hld_col_cnt_oth.append((col_name_oth, _col_elm_cnt))

        # List to DataFrame
        df_col_elem_cnt_oth = pd.DataFrame(list_hld_col_cnt_oth, columns = ['col_name', 'unq_val_cnt'])
        
        # Count all isnull
        srs_col_isnull = pd.DataFrame(indata[list_col_oth].isnull().sum(axis = 0)).rename(columns = {0 : 'isnull_cnt'}).reset_index(drop = True)
        
        # Column names and datatypes
        df_col_dtype_oth = pd.DataFrame(indata.dtypes).reset_index().rename(columns = {'index' : 'col_name', 0 : 'col_type'})
        df_col_dtype_oth = df_col_dtype_oth[df_col_dtype_oth['col_name'].isin(list_col_oth)].set_index('col_name')

        
        
        # Merge back count isnull + unique values and have correct order on columns
        df_col_elem_cnt_oth = pd.concat([df_col_elem_cnt_oth[['col_name']], srs_col_isnull,df_col_elem_cnt_oth[['unq_val_cnt']]]
                                   ,axis = 1, sort=False)

        # Count total nr rows
        df_col_elem_cnt_oth['n_rows_tot'] = len(indata)
        
        # Calculate % missing values for given column
        df_col_elem_cnt_oth['%_null_tot'] = df_col_elem_cnt_oth['isnull_cnt'].div(df_col_elem_cnt_oth['n_rows_tot'])

        # Copy 
        df_cols_meta_oth = df_col_elem_cnt_oth.copy()


        # Describe on all input 
        df_describe = indata[df_cols_meta_oth.col_name.tolist()].describe().T

        # Set index to column col_name in meta table to enable join on index
        df_cols_meta_oth.set_index('col_name', inplace=True)

        # Merge with metadata, exclude index and count
        df_cols_meta_oth = pd.concat([df_cols_meta_oth,df_col_dtype_oth, df_describe[[col for col in df_describe.columns if col not in (['count'])]]]
                                ,axis = 1, sort=False).reset_index()

        # Final fix + Sort order
        df_cols_meta_oth.rename(columns={'index' : 'col_name'}, inplace = True)                
        df_cols_meta_oth = df_cols_meta_oth[['col_name', 'col_type'] + [col for col in df_cols_meta_oth.columns if col not in (['col_name','col_type'])]]
        
        

        #-----------------------------------------------------------------
        # Here we return numeric output and other output (string + date)
        #-----------------------------------------------------------------
        return df_cols_meta_num, df_cols_meta_oth
        
    else:
        
        print ("No execution of table overview, ending...")
        return None, None
        
        
 
        

def f_row_count(exec_f,indata,desc_str,row_count,id_count,id_count_notn,disc_vc):


    """
    Function for printing out a row overview of a given table object (Pandas data structre, e.g.)
    
    Parameters:
    
    exec_f                   Boolean (True/False): Either execute, or not.
    table                     Input table object to do count on, containing id variables
    desc_str                String; input on description on table counting on
    row_count             Bolean (True/False): Prints out number of rows on the table object 'table'
    id_count                List of column names, as string, else False: prints out distinct count of a given list of ids
    id_count_notnull     List of column names, as string, else False: prints out distinct count of a given list of ids with condition not null applied
    disc_vc                  List of column names, as string, else False: print out a value counts overview of discrete data in list, i.e. segments, etc.
    
    """

    if exec_f:
        import pandas as pd
        import numpy as np


        print ("----------------------------------------------------------------------------------")
        print ("Row overview - %s: " % desc_str)
        print ("----------------------------------------------------------------------------------")


        # All rows count
        if row_count:
            print ("%s %s\n" % ('{:45s}'.format('N rows:'),len(indata)))
  

        if id_count:
            # No dup and count on list input id variables
            for idx, var in enumerate(id_count):

                print ("%s %s\n" % ('{:45s}'.format('N no-dup rows ' + var + ':')
                                  ,len(indata[var].drop_duplicates())
                                 )
                      )


        if id_count_notn:
            # No dup and count on list input id variables where rows are not null
            for idx, var in enumerate(id_count_notn):
                print ("%s %s\n" % ('{:45s}'.format('N no-dup not null rows ' + var + ':')
                                  ,len(indata[indata[var].notnull()][var].drop_duplicates())
                                 )
                      )



        if disc_vc:
            # Value counts to get distribution of discrete variables
            for idx_d, var_d in enumerate(disc_vc):
                print ("------------------------------------")
                print ("N rows dist. var. '%s': " % var_d)
                print ("------------------------------------")
                print (indata[var_d].value_counts().sort_index())    
                print ("\n")

    else:

        print ("No execution of row count function, passing input data")
        return indata


def f_clean_text(exec_f
                 ,text_var_in
                 ,text_var_label):
    
    """
    Function for applying regular expression to clean out signs, e.g. (?=`^*), etc. The reg-ex saves emotions like :), =), etc...
    Output is cleaned text, in a data frame form
    
    text_var_in         List containing documents as strings, to be cleaned
    text_var_label      Label on output vector with cleaned text
    
    """
    # Execute function with cleaning
    if exec_f:
        import re
        import pandas as pd
        
        # Empty holder to encapsulate cleaned
        holder = list()
        
        # Loop to go through list with text
        for idx, val in enumerate(text_var_in):
            
            text = re.sub('<[^>]*>', '', val)
            emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
            text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
        
            # Append cleaned text
            holder.append(text)
            
        # To data frame for easy merge back to org file
        out_text = pd.DataFrame({text_var_label : holder})
            
        # Push back output
        return out_text

    # No execution of function
    else:
        print ("No cleaning performed, passing text....")            

        
        
def f_brdc_grpby(exec_f
                  ,indata
                  ,list_grp_by_var
                   ,grp_by_arg
                 
                  ,lgc_var
                  ,lgc_expr
                  ,lgc_name_out):
    
    
    """
    
    Docstring for groupby and logical value broadcasting when creating a new variable. 
    Function replaces use of transform step with apply function where time of execution is much too long.
    
    NOTE: Fill missing values given left-join has to be done manually to avoid data type mixing 
    
    Parameters:
    
    exec_f                   Boolean (True/False):  Execute function or not
    indata                   Dataframe: Holding data to be processed
    
    list_grp_by_var          List: Name(s) of variable(s) that groupby object is based on
    grp_by_arg               String: Name of group-by operator call (count, sum, first, etc...)
    
    lgc_var                   String: Name of variable that logical expression is applied on    
    lgc_expr                  String or False: If False, no subselect is done on data, else a query (see: help(pd.DataFrame.query) for details) string to subset dataframe
    lgc_name_out              String: Name of the variable being created
    
    """
    
    
    # Do we execute the function, yes/no
    if exec_f:
        print ("Executing groupby and aggregation for logical expression subset:")
        

        # Logical equal too False, we dont subset data through query and only create new var based on grp_by_arg
        if lgc_expr == False:

            print ("Executing function without subset selection: %s \n" % (lgc_expr
                                                                        )
                  )
            
            # Hold columns to be processed
            tmp_lst = list(list_grp_by_var)
            tmp_lst.append(lgc_var)
            

            # Initialize group-by object for given logical variable, expression and value and a given aggregation function
            _grp_by = indata[tmp_lst].groupby(list_grp_by_var, as_index = False).agg(grp_by_arg)
          
            # Return object with broadcasted aggregation from groupby object attached to dataframe
            return indata.merge(_grp_by.rename(columns = {lgc_var : lgc_name_out})
                                ,how = 'left'
                                ,left_on = list_grp_by_var
                                ,right_on = list_grp_by_var)
            
        
        elif type(lgc_expr) == str:
        
            print ("Executing logical function: %s \n" % (lgc_expr
                                                       )
                  )
            
            
            # Hold columns to be processed
            tmp_lst = list(list_grp_by_var)
            tmp_lst.append(lgc_var)

                

            # Initialize group-by object for given logical variable, expression and value and a given aggregation function
            _grp_by = indata.query(lgc_expr)[tmp_lst].groupby(list_grp_by_var, as_index = False).agg(grp_by_arg)
          
            # Return object with broadcasted aggregation from groupby object attached to dataframe
            return indata.merge(_grp_by.rename(columns = {lgc_var : lgc_name_out})
                                ,how = 'left'
                                ,left_on = list_grp_by_var
                                ,right_on = list_grp_by_var)

        
        else:
            print ("Need to specify a logical expression to subset data, exiting....")

    # No function exeuction   
    else:
        print ("No execution of function, ending....")



def f_vc_aug(exec_f
             ,indata
             ,var
             ,plot = True
             ):             
    """
    Function for doing an augmenation on the value counts function from Pandas with addition of 
    % of total and cummulative % on the discrete variable the calculation is applied on. The function
    also pushes two plots.
    
    Parameters:
    
    exec_f            Boolean(True/False). Execute function or not
    indata            pandas DataFrame object (n*m): DataFrame containing variable to make calculation on.
    var               string (n*1): String name of variable to make calculation on.
    plot              Boolean(True/False): Execute plot or not.
    
    """

    if exec_f == True:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns


        df = pd.DataFrame(indata[var].value_counts().sort_index())


        df['%_of'] = df.div(df[var].sum())

        df['%_of_cumsum'] = df['%_of'].cumsum()
            
        df['%_of'][:25].sum()
        
        df.sort_values(by = '%_of_cumsum', inplace = True)
        
        df.reset_index(inplace  = True)
        
        df.rename(columns = {'index' : 'category'}, inplace = True)

        
        # Plot dist
        if plot:
            fig_idx_plt, ax_idx_plt = plt.subplots(figsize = (15, 6))
            sns.barplot(x="category", y=var, data=df)
            plt.show()
                
            
        
        return df
            
            
    else:
        print ("No exeuction of function, ending...")        
        
        
def f_srt_dict(exec_f
              ,lst
              ,diction
              ,value):
    
    """
    Docstring for srt dict function:
    
    Function sorts either two list being merged to a dictionary (zip), or an existing dictionary, by either key or value.
    
    Parameters:
    
    exec_f       Boolean(True/False): If True, function is executed, else not
    
    lst         List/False: List with 2 lists which will become KEYS and VALUES in the dictionary
                
    diction      Dictionary/False: If defined, dictionary will be sorted and returned according to value parameter
    value        Boolean(True/False): If True, dictionary or zip(list1, list2) will be sorted by VALUE, if False, sorted by KEY
    
    
    """
    
    if exec_f:
    
        if (lst) and diction == False:
    
            _temp = dict(zip(lst[0]
                             ,lst[1]
                            )
                        )
        
            if value == True:
        
                _temp_s = sorted(_temp.items(),key = operator.itemgetter(1))
            
            else:
                
                _temp_s = sorted(_temp.items(),key = operator.itemgetter(0))

            return _temp_s
        
        elif (lst == False) and diction:
            
            if value == True:
            
                return sorted(diction.items(),key = operator.itemgetter(1))
            
            else:
        
                return sorted(diction.items(),key = operator.itemgetter(0))
        
        else:
            
            print ("No sort input, returning None")
            
            return None
        
    else:
        
        print ("No execution of function, ending...")
    
    
        
def f_fillna(exec_f
            ,indata
            ,col_fillna
            ,val_fillna
            ,flag_fillna):
    
    """
    Function for filling a missing value
    
    Parameters:
    
    exec_f        Boolean (True/False): Execute function or not. If not, return indata
    
    indata        Object, Pandas DataFrame: Input data containing the columns with NaN to be filled
    col_fillna    List of strings: Name(s) of column(s) to have value(s) filled
    val_fillna    Numeric/String: Value to fill column
    flag_fillna   Boolean (True/False): if True, flag is created which sets 1 if value is imputed, else not
    
    
    """
    
    # Go/No-Go
    if exec_f:
        import pandas as pd
        import numpy as np

        #-----------------------------------------------------------------------
        # We flag the imputed rows, i.e. NaN --> f_imp_"col name" == 1, else 0
        #-----------------------------------------------------------------------
        if flag_fillna:
            
            # We have one column only
            if len(col_fillna) == 1:

                # Flag NaN rows first, to be imputed
                indata['f_imp_' + col_fillna[0]] = np.where(indata[col_fillna[0]].isnull()
                                                     ,1
                                                     ,0)

                # Then we impute data
                indata[col_fillna[0]] = indata[col_fillna[0]].fillna(val_fillna)
                
                return indata
                
            # We have more than one column to be filled, loop through each column and set flag accordingly
            else:
                
                # Initialize for-loop to set flags;
                for idx_col, col in enumerate(col_fillna):
                
                    # Flag NaN rows first, to be imputed
                    indata['f_imp_' + col] = np.where(indata[col].isnull()
                                                             ,1
                                                             ,0)

                # Then we impute data
                indata[col_fillna] = indata[col_fillna].fillna(val_fillna)
                
            
            return indata

        #------------------------------------
        # No flag of imputed rows
        #------------------------------------
        else:
           
            # Only one col to be imputed
            if len(col_fillna) == 1:
            
                # Then we impute data
                indata[col_fillna[0]] = indata[col_fillna[0]].fillna(val_fillna)
                
            # List of cols to be imputed
            else:                
                # We impute data
                indata[col_fillna] = indata[col_fillna].fillna(val_fillna)
                
            return indata
                
    else:
        
        print ("No execuction of impute function, ending. Returning indata")
        return indata    
        

        
        



class c_atlas(object):
    
    """
    Description:
    
    Version 1 of Atlas, heavy lifting program for doing pivot table aggregation that gets loaded into excel and exported for
    exploratory analysis.
    
    Modules:
    
    import pandas as pd
    from sys import exit
    
    Class assignment
    
    indata            DataFrame - Data holding information to be aggregated. Columns used as aggregators, i.e. attributes, and numerical column that gets summed up. 
                      Also, one "id column" with unique values represeting the frequency (number of rows) in the data
                      
    Parameters:
                      
            
    agg_col           List of string names represeting columns - Discrete attributes used to group the data on
    sum_col           List of string names of columns to be summed up, else False and only count is done
    freq_col          String name represeting column to be used for rows frequency
    fillna            Boolean (True/False), apply fillna to column
    fillna_val        If fillna True, provide value, else False
                    
    Output control parameters:
    
    out_csv           Boolean, True/False, if True then write out csv file
    path              String output path of to push aggregated DataFrame in csv form
    file              String name, with file ending
    return_agg        Boolean, True/False, if True return agg D.F, else only push agg --> CSV
    
    """
    
    
    # Define init and indata, and datestamt for output tag
    def __init__(self, indata):
        
        self.indata = indata
        
    # Atlas, agg
    def aggregate(self
                 ,agg_col
                 ,sum_col
                 ,freq_col
                 ,fillna
                 ,fill_val
                 ,out_csv
                 ,path
                 ,file
                 ,return_agg):
        
        # Self
        self.agg_col = agg_col
        self.sum_col = sum_col
        self.freq_col = freq_col
        self.fillna = fillna
        self.fill_val = fill_val
        self.out_csv = out_csv
        self.path = path
        self.file = file
        self.return_agg = return_agg
        
        # Declare holder of output from indata aggregation
        self.hold_agg = pd.DataFrame()
        
        # Check instance of indata, if not pd.DataFrame then raise error and exit, else proceed
        if isinstance(self.indata, pd.core.frame.DataFrame):

            # Summary col exist
            if sum_col:
            
                self.sum_df = self.indata[self.agg_col + self.sum_col].groupby(self.agg_col).sum()
                self.freq_df = self.indata[self.agg_col + self.freq_col].groupby(self.agg_col).count()

                # First level of agg is summation on numeric input, grouped by aggregation attributes
                # Second aggregation is a row level count, group by aggregation attributes
                # Concat over axis 1
                self.hold_agg = pd.concat([self.sum_df
                                          ,self.freq_df]
                                          ,axis = 1, sort=False)

                # Rename columns going in with prefix "s_" and count to "freq_"
                self.hold_agg.rename(columns = lambda i: 's_' + i if i not in (self.agg_col + self.freq_col) else 'freq_' + i
                                ,inplace = True)

                
                # Fill missing value with assigned scalar, function, dictionary, etc...
                if self.fillna:
                    self.hold_agg.fillna(self.fill_val,
                                 inplace = True)
              
            # We only look at the frequency
            else:
                
                self.freq_df = self.indata[self.agg_col + self.freq_col].groupby(self.agg_col).count()

                # First level of agg is summation on numeric input, grouped by aggregation attributes
                # Second aggregation is a row level count, group by aggregation attributes
                # Concat over axis 1
                self.hold_agg = self.freq_df
                

                # Rename columns going in with prefix "s_" and count to "freq_"
                self.hold_agg.rename(columns = {str(self.freq_col).strip('[]').strip("'") : 'freq_' + str(self.freq_col).strip('[]').strip("'")}
                                ,inplace = True)


                
                # Fill missing value with assigned scalar, function, dictionary, etc...
                if self.fillna:
                    self.hold_agg.fillna(self.fill_val
                                        ,inplace = True)
                
                
            
            # We push out output to csv for EDA through pivot in excel
            if self.out_csv:
                
                # Add date tag to ouput file
                self.today = dt.datetime.today().strftime("%y%m%d")
                
                self.hold_agg.reset_index(inplace = True)
                
                # Output file through parameters
                self.hold_agg.to_csv(self.path + self.today + '_' + self.file
                             ,sep = ';'
                             ,decimal = ','
                             ,index = False)
                
                # Also, return DF
                if self.return_agg:
                    
                    # return
                    return self.hold_agg
                
            # Only return DataFrame
            else:
                
                # return
                return self.hold_agg
            
            
        # Indata not pd.DataFrame, exist...
        else:
            print ("Indata file not of type pd.core.frame.DataFrame, ending....")
            exit()

