# Characteristics Kernel Principal Component Analysis (CK-PCA)

This repository replicates CK-PCA, a non-linear version of PCA and a modification of Kernel PCA (K-PCA), introduced by **Serhiy Kozak** in his paper **Kernel Trick for the Cross Section**.

[Link to the original paper on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3307895).

## Class API
CK-PCA class follows the API naming conventions of Sklearn. The class possesses a fit function, along with a transform function to retrieve in-sample and out-of-sample PCs. 

# Example Usage

First, define a set of parameters.

```python
parameters = {
    'data_source': 'OSAP',
    'kernel': 'linear',
    'c': np.nan,
    'ultimate_sample_start_date': '2015-12-01',
    'sample_start_date': '2015-12-01',
    'oos_split_date': '2022-11-01',
    'sample_end_date': '2022-12-31',
    'ultimate_sample_end_date': '2022-12-31',
    'freq_rets': 'M',
    'characteristics_model': 'FF5',
    'path_cache': 'omega_cached',
}

```

CK-PCA's input consists of stock returns along with rank-transformed and normalized characteristic singals. 

To load the data, run:

```python
asset_rets, characteristics = load_characteristics_data_from_cache(parameters)

characteristics_is, characteristics_oos = train_test_split_double_index(characteristics, parameters['oos_split_date'])
asset_rets_is, asset_rets_oos = train_test_split_double_index(asset_rets, parameters['oos_split_date'])

```

Next, initialize the class:

```python
ckpca = CKPCA(
    kernel=parameters['kernel'],
    c=parameters['c'],
    characteristics_model=parameters['characteristics_model'],
    data_source=parameters['data_source'],
    storage='local',
    path_cache='omega_cached',
)

```

Next, fit the class with the sample data. Note that the full sample of the characteristics and return data is inputted in the fit function. With that, CK-PCA computes the full sample Omega-matrix, from which in-sample and out-of-sample PCs can be cheaply computed.

```python
ckpca.fit(characteristics, asset_rets)

```
Therefore, the transform function takes only one input argument, the out-of-sample split date.
```python
ck_pcs_is, ck_pcs_oos = ckpca.transform(split_date=parameters['oos_split_date'])

```