# %%
import pandas as pd
import numpy as np
from ckpca.helpers import load_characteristics_data_from_cache, train_test_split_double_index, rotate_rets
from ckpca.CKPCA import CKPCA

parameters = {
    'data_source': 'OSAP', # 'kozak' or 'OSAP'
    'kernel': 'poly2',
    'c': 10**1,
    'ultimate_sample_start_date': '2015-12-01',
    'sample_start_date': '2015-12-01',
    'oos_split_date': '2022-11-01',
    'sample_end_date': '2022-12-31',
    'ultimate_sample_end_date': '2022-12-31',
    'freq_rets': 'M',
    'characteristics_model': 'Clean',
    'path_cache': 'omega_cached',
}

# %%
asset_rets, characteristics = load_characteristics_data_from_cache(parameters)

characteristics_is, characteristics_oos = train_test_split_double_index(characteristics, parameters['oos_split_date'])
asset_rets_is, asset_rets_oos = train_test_split_double_index(asset_rets, parameters['oos_split_date'])

# %%
##%%time

ckpca = CKPCA(
    kernel=parameters['kernel'],
    c=parameters['c'],
    characteristics_model=parameters['characteristics_model'],
    data_source=parameters['data_source'],
    storage='local',
    path_cache=parameters['path_cache'],
    freq=parameters['freq_rets'],
)

ckpca.fit(characteristics, asset_rets)

ck_pcs_is, ck_pcs_oos = ckpca.transform(split_date=parameters['oos_split_date'])

pcs_full_sample = pd.concat([ck_pcs_is, ck_pcs_oos], axis=0)

# %% Compare CK-PCs to PCs from standard PCA
lin_rets_is =  characteristics_is.multiply(asset_rets_is, axis=0)
lin_rets_is = lin_rets_is.groupby('date').sum()
lin_rets_is.name = 'lin_ret'

lin_rets_oos = characteristics_oos.multiply(asset_rets_oos, axis=0)
lin_rets_oos = lin_rets_oos.groupby('date').sum()
lin_rets_oos.name = 'lin_ret'

lin_pcs_is = rotate_rets(lin_rets_is)
lin_pcs_oos = rotate_rets(lin_rets_oos, lin_rets_is)

lin_pcs_full_sample = pd.concat([lin_pcs_is, lin_pcs_oos], axis=0)
# %%
def get_corrs(df1, df2):
    corrs = pd.DataFrame()
    for n in range(123):
        try:
            #print(n, np.abs(lin_pcs_oos.iloc[:, n].corr(pcs_oos.iloc[:, n])))
            print(n, np.abs(df1.iloc[:, n].corr(df2.iloc[:, n])))
            corrs.loc[n, 'corr'] = np.abs(df1.iloc[:, n].corr(df2.iloc[:, n]))
            #corrs.loc[n, 'corr'] = np.abs(lin_pcs_oos.iloc[:, n].corr(pcs_oos.iloc[:, n]))
        except:
            pass

    corrs.plot(ylim=[0, 1.1], title=f'Correlation of CK-PCs ({parameters["kernel"]} kernel) and linear PCs for {parameters["data_source"]}/{parameters["characteristics_model"]} with large c.')

get_corrs(lin_pcs_full_sample, pcs_full_sample)

# %%
