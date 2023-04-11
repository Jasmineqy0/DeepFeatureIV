from collections import defaultdict
import plotly.express as px
from glob import glob
import os
import pandas as pd
import json
import csv

DUMP_DIR = '../dumps'
# model_name: 
dump = {'dfiv': '04-11-01-35-35',}

alldict = defaultdict(list)
for key, dir in dump.items():
    m = 0
    for subdir in glob(os.path.join(DUMP_DIR, dir, 'data_size*')):
        params = [param.split(':') for param in subdir.split('/')[-1].split('-')]
        
        if os.path.exists(os.path.join(subdir, 'result.csv')):
            with open (os.path.join(subdir, 'result.csv'), 'r') as f:
                reader = csv.reader(f)
                mse = [float(row[0]) for row in list(reader)]
                n = len(mse)
                m += n

            params = {param[0]: [param[1]] * n for param in params}
            subdict = {**params, 'mse': mse, 'model': [key] * n}
            alldict = {key: subdict[key] + alldict[key] for key in subdict.keys()}

df = pd.DataFrame(alldict)

fig = px.box(df, x='data_size', y='mse', color='model', log_y=True,
                facet_col='rho', facet_col_wrap=5)
fig.show()

print(df)

    
        