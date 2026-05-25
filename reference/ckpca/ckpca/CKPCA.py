import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import polynomial_kernel, sigmoid_kernel
from joblib import Parallel, delayed
from typing import Union
from tqdm import tqdm

class CKPCA(object):
    def __init__(self, kernel, c, characteristics_model, data_source, storage, path_cache, freq='M'):

        self.kernel = kernel
        self.c = c
        self.characteristics_model = characteristics_model
        self.data_source = data_source
        self.storage = storage
        self.path_cache = path_cache
        self.freq = freq


    @staticmethod
    def get_eigenvalues_eigenvectors(
        omega: pd.DataFrame, 
        rescale: bool = True, 
        plot: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Computes the eigenvalues and eigenvectors of the given omega matrix.
    
        This method calculates the eigenvalues and eigenvectors of the omega matrix,
        sorts the eigenvalues in descending order, and adjusts the eigenvectors accordingly.
        Optionally, it can plot the eigenvalues and rescale the eigenvectors.
    
        Args:
            omega (pd.DataFrame): The omega matrix for which to compute eigenvalues and eigenvectors.
            rescale (bool): Whether to rescale the eigenvectors by the square root of the eigenvalues.
            plot (bool): Whether to plot the eigenvalues.
    
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - eigenvalues (pd.DataFrame): The sorted eigenvalues.
                - eigenvectors (pd.DataFrame): The corresponding eigenvectors.
        """

        # Perform eigenvalue decomposition on omega
        eigenvalues, eigenvectors = np.linalg.eigh(omega)

        # Convert eigenvectors and eigenvalues to DataFrames
        eigenvectors = pd.DataFrame(eigenvectors)
        eigenvalues = pd.DataFrame(eigenvalues)
    
        eigenvalues = eigenvalues.sort_values(by=0, ascending=False)
    
        # Reverse the order of the columns in eigenvectors
        eigenvectors = (
            eigenvectors
            .iloc[:, ::-1]
            .iloc[:, :eigenvalues[eigenvalues[0] > 0].shape[0]]
        )
    
        # Rename columns of eigenvectors to PC1, PC2, ...
        eigenvectors.columns = [f'PC{i+1}' for i in range(eigenvectors.shape[1])]
        eigenvectors.index = omega.index
        eigenvalues.index = [f'PC{i+1}' for i in range(eigenvalues.shape[0])]
    
        eigenvalues = eigenvalues[eigenvalues[0] > 0]
    
        if plot:
            eigenvalues.plot()
    
        if rescale:
            eigenvectors = eigenvectors * np.sqrt(eigenvalues[0])
    
        return eigenvalues, eigenvectors

    @staticmethod
    def center_kernel(K: pd.DataFrame) -> pd.DataFrame:
        """
        Centers the kernel matrix by subtracting the row and column means and adding the overall mean.
    
        Args:
            K (pd.DataFrame): The kernel matrix to be centered.
    
        Returns:
            pd.DataFrame: The centered kernel matrix.
        """
        row_means = K.mean(axis=1, keepdims=True)
        col_means = K.mean(axis=0, keepdims=True)
        all_mean = K.mean()
        centered_K = (
            K 
            - row_means 
            - col_means 
            + all_mean
        )
        return centered_K

    @staticmethod
    def calculate_centered_kernel(
        X1: pd.DataFrame, 
        X2: pd.DataFrame, 
        kernel_type: str = 'polynomial', 
        c: float = 1.0
    ) -> pd.DataFrame:
        """Calculates the centered kernel matrix for the given data and kernel type.
    
        Args:
            X1 (pd.DataFrame): The first data matrix.
            X2 (pd.DataFrame): The second data matrix.
            kernel_type (str): The type of kernel to use ('poly2', 'rbf', 'sigmoid', 'linear').
            c (float): The kernel coefficient parameter.
    
        Returns:
            pd.DataFrame: The centered kernel matrix.
        """
        if kernel_type == 'poly2':
            K = polynomial_kernel(X1, X2, coef0=c, degree=2, gamma=1)
        elif kernel_type == 'rbf':
            K = RBF(length_scale=c)(X1, X2)
        elif kernel_type == 'sigmoid':
            K = sigmoid_kernel(X1, X2, gamma=c)
        elif kernel_type == 'linear':
            K = (X1 @ X2.T).to_numpy()
        
        return pd.DataFrame(CKPCA.center_kernel(K), index=X1.index, columns=X2.index)

    @staticmethod
    def compute_omega_element_for_date(
        characteristics: pd.DataFrame,
        asset_rets: pd.Series,
        date1: str,
        date2: str,
        kernel_type: str = 'linear',
        c: float = 1, 
        only_weights: bool = False
    ) -> Union[float, pd.Series]:
        """
        Computes the omega element for a given pair of dates.
    
        This function calculates the omega element for the specified dates using the provided
        characteristics and asset returns. Depending on the `only_weights` flag, it either returns
        a single value or a vector.
    
        Args:
            characteristics (pd.DataFrame): The characteristics data for the assets.
            asset_rets (pd.Series): The asset returns data.
            date1 (str): The first date for which to compute the omega element.
            date2 (str): The second date for which to compute the omega element.
            kernel_type (str, optional): The type of kernel to use. Defaults to 'linear'.
            c (float, optional): The kernel coefficient parameter. Defaults to 1.
            only_weights (bool, optional): If True, returns only the weights. Defaults to False.
    
        Returns:
            Union[float, pd.Series]: The computed omega element. If `only_weights` is True, returns
            a vector of weights. Otherwise, returns a single value.
        """
        X1 = characteristics.xs(date1, level=1)
        X2 = characteristics.xs(date2, level=1)
    
        centered_K = CKPCA.calculate_centered_kernel(X1, X2, kernel_type=kernel_type, c=c)
        
        r1 = asset_rets.xs(date1, level=1)
        r2 = asset_rets.xs(date2, level=1)
        
        if only_weights:
            w = r1.loc[centered_K.index] @ centered_K
            return w
        else:
            return r1.loc[centered_K.index] @ centered_K @ r2.loc[centered_K.columns]
        
    def process_date_combinations(
        self, 
        characteristics: pd.DataFrame, 
        asset_rets: pd.Series,
        date_combinations: list
    ) -> zip:
        """Initializes the computation of omega elements in parallel.

        Args:
            characteristics (pd.DataFrame): A DataFrame of characteristics data.
            asset_rets (pd.Series): A Series of asset returns data.
            date_combinations (list): A list of date combinations for which to compute omega elements.

        Returns:
            zip: A zip object containing the date combinations and the corresponding omega elements.
        """
        omega_elements = Parallel(n_jobs=-1, prefer="threads")(
            delayed(CKPCA.compute_omega_element_for_date)(
                characteristics, 
                asset_rets, 
                date1, 
                date2,
                kernel_type=self.kernel,
                c=self.c
            ) 
            for date1, date2 in tqdm(date_combinations, desc="Processing date combinations")
        )
        return omega_elements


    def init_omega(self):
        """Initializes the omega attribute based on the parameters provided.

        This method constructs the omega name string based on the kernel type and other parameters.
        If the kernel is 'linear', the omega name is constructed without the 'c' parameter.
        Otherwise, the 'c' parameter is included in the omega name.
        After constructing the omega name, it attempts to load the pre-calculated omega from the cache.

        Parameters:
        None

        Returns:
        None
        """

        if self.kernel == 'linear':
            self.omega_name = (
                f"omega_{self.kernel}_"
                f"{self.ultimate_sample_start_date}_"
                f"{self.ultimate_sample_end_date}_"
                f"{self.characteristics_model}_"
                f"{self.data_source}"
                f"{self.freq}"
            )
        else:
            self.omega_name = (
                f"omega_{self.kernel}_"
                f"{self.c}_"
                f"{self.ultimate_sample_start_date}_"
                f"{self.ultimate_sample_end_date}_"
                f"{self.characteristics_model}_"
                f"{self.data_source}"
                f"{self.freq}"
            )
    
        # Attempt to load the pre-calculated omega
        self.load_omega_from_cache()

    def load_omega_from_cache(self):
        """Attempts to load omega from the cache based on the storage type.
        """
        if self.storage == 'local':
            self.load_omega_from_cache_local()
        elif self.storage == 'cloud':
            raise NotImplementedError('Cloud storage not implemented yet.')
        
    def load_omega_from_cache_local(self):
        """Checks if the omega is already pre-computed and cached. 
        
        If it is, it loads the omega from the cache and sets fit attribute to True. Otherwise, 
        it sets the fit attribute to False.
        """
        try:
            self.omega = pd.read_pickle(f"{self.path_cache}/{self.omega_name}.p")
            self.fit = True
        except FileNotFoundError:
            self.fit = False

    def set_sample_dates(self, asset_rets: pd.Series):
        """This function sets two class parameters based on the asset returns data.

        Ultimate sample start (end) date are important, as they define the name of 
        the cached omega matrices.

        Args:
            asset_rets (pd.Series): Sample asset returns data.
        """
        sample_dates = asset_rets.index.get_level_values(1).unique().sort_values()
        self.ultimate_sample_start_date = sample_dates[0].strftime('%Y-%m-%d')
        self.ultimate_sample_end_date = sample_dates[-1].strftime('%Y-%m-%d')

    def fit(self, characteristics: pd.DataFrame, asset_rets: pd.Series):
        """The fit method for the CK-PCA class serves as the main entry point for the class. 
        
        It initializes class parameters and checks if omega is already pre-computed and cached.
        If omega is not pre-computed, it calls the function to compute omega.

        Args:
            characteristics (pd.DataFrame): A DataFrame of characteristics data.
            asset_rets (pd.Series): A Series of asset returns data.
        """

        # Set the sample dates based on the asset returns
        self.set_sample_dates(asset_rets)

        # Initializes omega name and checks if omega is already pre-computed
        self.init_omega()
        
        if not self.fit:
            print('Fitting CK-PCA...')
            self.compute_omega(characteristics, asset_rets)

        print('CK-PCA fit complete.')

    def compute_omega(self, 
        characteristics: pd.DataFrame, 
        asset_rets: pd.Series
    ):
        """Main function of the class. It computes the omega matrix for CK-PCA.

        This function initializes omega as a DataFrame with date indices. It then computes the omega
        matrix by iterating over all possible date combinations, computing the omega element for each
        combination, and populating the omega DataFrame.

        Args:
            characteristics (pd.DataFrame): A DataFrame of characteristics data.
            asset_rets (pd.Series): A Series of asset returns data.
        """
        
        # Get the unique dates in the sample
        sample_dates = asset_rets.index.get_level_values(1).unique().sort_values()
        
        # init omega as a DataFrame
        self.omega = pd.DataFrame(index=sample_dates, columns=sample_dates)
        
        # Create unique date combinations where d1 <= d2
        date_combinations = [(d1, d2) for i, d1 in enumerate(sample_dates) for d2 in sample_dates[i:]]
        
        # parallel computation of omega via joblib
        omega_elements = self.process_date_combinations(characteristics, asset_rets, date_combinations)
        
        # Populate the omega DataFrame with the chunk results
        for (date1, date2), value in zip(date_combinations, omega_elements):
            self.omega.loc[date1, date2] = value
            self.omega.loc[date2, date1] = value
        
        self.omega = self.omega.astype(float)
        
        # cache omega to pickle file
        self.omega.to_pickle(f"{self.path_cache}/{self.omega_name}.p")
        
        self.fit = True
    
    def transform(self, split_date: str):
        """Calls functions to compute CK-PCs for IS and OOS period, based on the provided split date.

        Contrary to typical sklearn transform methods, this method does not take any input data. This
        is due to the fact that once the class is fitted, only omega is required to compute CK-PCs.

        Args:
            split_date (str): The date used to split the data into in-sample and out-of-sample sets.

        Returns:
            tuple: A tuple containing two DataFrames:
                - ck_pcs_is (pd.DataFrame): The principal components for the in-sample data.
                - ck_pcs_oos (pd.DataFrame): The principal components for the out-of-sample data.
        """
        
        ck_pcs_is = self.get_ck_pcs_is(split_date)

        ck_pcs_oos = self.get_ck_pcs_oos(split_date)

        return ck_pcs_is, ck_pcs_oos


    def get_ck_pcs_is(self, split_date: str):
        """Computes the CK-PCs for the in-sample period based on the provided split date.

        This function filters the omega matrix to only include dates before the split date. It then
        computes the eigenvectors of the filtered omega matrix and returns the scaled eigenvectors as
        the CK-PCs for the in-sample period.

        Args:
            split_date (str): The date used to split the data into in-sample and out-of-sample sets.

        Raises:
            ValueError: If the eigenvectors are empty, an error is raised.

        Returns:
            pd.DataFrame: The CK-PCs for the in-sample period.
        """
        
        # filter omega to only include dates before the split date
        is_row_indices = self.omega.index.get_level_values(0) <= split_date
        is_col_indices = self.omega.index.get_level_values(0) <= split_date
        omega_is = self.omega.loc[is_row_indices, is_col_indices]

        # CK-PCA: Scaled eigenvectors of omega correspond to CK-PCs
        _, ck_pcs_is = CKPCA.get_eigenvalues_eigenvectors(omega_is, rescale=True, plot=False)

        # if self.eigenvectors is empty, throw an error
        if ck_pcs_is.empty:
            raise ValueError('Failed to compute eigenvectors and eigenvalues.')
        
        return ck_pcs_is


    def get_ck_pcs_oos(self, split_date: str):
        """Computes the CK-PCs for the out-of-sample period based on the provided split date.

        First, this function filters the omega matrix to only include dates before the split date. This
        matrix is used to compute scaling factors for the out-of-sample omega matrix. This matrix is
        also a part of full sample omega matrix.

        Args:
            split_date (str): The date used to split the data into in-sample and out-of-sample sets.

        Returns:
            pd.DataFrame: The CK-PCs for the out-of-sample period.
        """
        
        # filter omega to only include dates before the split date
        is_row_indices = self.omega.index.get_level_values(0) <= split_date
        is_col_indices = self.omega.index.get_level_values(0) <= split_date
        omega_is = self.omega.loc[is_row_indices, is_col_indices]

        # CK-PCA: Eigenvectors and lamba are used to weight the out-of-sample data
        lambdas, alphas = CKPCA.get_eigenvalues_eigenvectors(omega_is, rescale=True, plot=False)

        # filter omega to only include cols after the split date
        oos_col_indices = self.omega.index.get_level_values(0) > split_date
        omega_oos = self.omega.loc[is_row_indices, oos_col_indices]
        
        # compute CK-PCs for the out-of-sample period
        ck_pcs_oos = lambdas[0]**(-1)*((alphas.T @ omega_oos).T)
        
        return ck_pcs_oos
    