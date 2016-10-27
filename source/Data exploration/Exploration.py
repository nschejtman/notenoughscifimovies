import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools as itls
from openpyxl import load_workbook


def read_items():
    items_df = pd.read_csv('../../inputs/item_profile.csv', sep='\t')
    return items_df


def write_description(data, ex_wrtr):
    print "Printing some statistics of the data\n"
    #print data.describe(include='all')
    data.describe(exclude=['object']).to_excel(ex_wrtr, sheet_name='Numerical_Description', na_rep='NA')
    data.describe(include=['object']).to_excel(ex_wrtr, sheet_name='Categorical_Description', na_rep='NA')
    ex_wrtr.save()


items_df = read_items()
out_name = 'output.xlsx'
ex_wrtr = pd.ExcelWriter(out_name, engine='openpyxl')
try:
    ex_wrtr.book = load_workbook(out_name)
    ex_wrtr.sheets = dict((ws.title, ws) for ws in ex_wrtr.book.worksheets)
except IOError:
    pass

write_description(items_df, ex_wrtr)
