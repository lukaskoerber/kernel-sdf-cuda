import cupy as cp
from numba import cuda
import numpy as np
import pandas as pd
from typing import Union
from tqdm import tqdm

# Enable memory pool
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

class CKPCA(object):
    def __init__(self, kernel, c, characteristics_model, data_source, storage, freq, path_cache):

        self.kernel = kernel
        self.c = c
        self.characteristics_model = characteristics_model
        self.data_source = data_source
        self.storage = storage
        self.freq = freq
        self.path_cache = path_cache

    @staticmethod
    def compute_linear_kernel(X1_device, X2_device):
        return cp.dot(X1_device, X2_device.T)

    @staticmethod
    def compute_polynomial_kernel(X1_device, X2_device, degree, coef0):
        linear_kernel = cp.dot(X1_device, X2_device.T)
        return (linear_kernel + coef0) ** degree

    @staticmethod
    def compute_rbf_kernel(X1_device, X2_device, length_scale):
        """Compute the RBF kernel."""
        X1_sq_norms = cp.sum(X1_device ** 2, axis=1).reshape(-1, 1)
        X2_sq_norms = cp.sum(X2_device ** 2, axis=1).reshape(1, -1)
        sq_dists = X1_sq_norms + X2_sq_norms - 2 * cp.dot(X1_device, X2_device.T)
        K_device = cp.exp(-sq_dists / (2 * length_scale ** 2))
        return K_device

    @staticmethod
    def calculate_centered_kernel_cuda(X1_device, X2_device, kernel_type: str, c: float, stream) -> np.ndarray:
        with stream:
            if kernel_type == 'linear':
                K_device = CKPCA.compute_linear_kernel(X1_device, X2_device)
            elif kernel_type == 'poly2':
                K_device = CKPCA.compute_polynomial_kernel(X1_device, X2_device, degree=2, coef0=c)
            elif kernel_type == 'rbf':
                K_device = CKPCA.compute_rbf_kernel(X1_device, X2_device, length_scale=c)
            else:
                raise ValueError(f"Unsupported kernel type: {kernel_type}")

            row_means = cp.mean(K_device, axis=1, keepdims=True)
            col_means = cp.mean(K_device, axis=0, keepdims=True)
            all_mean = cp.mean(K_device)
            centered_K_device = K_device - row_means - col_means + all_mean

        return centered_K_device

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

    def compute_omega_element_for_date_cuda(
        self,
        characteristics_device: dict,
        asset_rets_device: dict,
        date1: str,
        date2: str,
        kernel_type: str = 'linear',
        c: float = 1,
        only_weights: bool = False
    ) -> Union[float, pd.Series]:

        stream = cp.cuda.Stream(non_blocking=True)  # Create a non-blocking stream

        X1_device = characteristics_device[date1]
        X2_device = characteristics_device[date2]
        centered_K_device = CKPCA.calculate_centered_kernel_cuda(X1_device, X2_device, kernel_type=kernel_type, c=c, stream=stream)

        r1_device = asset_rets_device[date1]
        r2_device = asset_rets_device[date2]

        with stream:
            if only_weights:
                w_device = cp.dot(r1_device, centered_K_device)
                result = cp.asnumpy(w_device)
            else:
                omega_device = cp.dot(cp.dot(r1_device, centered_K_device), r2_device)
                result = cp.asnumpy(omega_device)

        stream.synchronize()  # Synchronize the stream to ensure all operations are completed
        return result

    def process_date_combinations_cuda(
        self,
        characteristics_device: dict,
        asset_rets_device: dict,
        date_combinations: list,
        batch_size: int = 100  # Process in batches to control memory usage
    ) -> zip:
        """Initializes the computation of omega elements in parallel with CUDA, with a progress bar.

        Args:
            characteristics (pd.DataFrame): A DataFrame of characteristics data.
            asset_rets (pd.Series): A Series of asset returns data.
            date_combinations (list): A list of date combinations for which to compute omega elements.
            batch_size (int): The number of date combinations to process in each batch.

        Returns:
            zip: A zip object containing the date combinations and the corresponding omega elements.
        """
        omega_elements = []

        for batch_start in tqdm(range(0, len(date_combinations), batch_size), desc="Processing batches"):
            batch_combinations = date_combinations[batch_start:batch_start + batch_size]
            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(len(batch_combinations))]
            results = []

            for stream, (date1, date2) in zip(streams, batch_combinations):
                result = self.compute_omega_element_for_date_cuda(
                    characteristics_device,
                    asset_rets_device,
                    date1,
                    date2,
                    kernel_type=self.kernel,
                    c=self.c
                )
                results.append(result)

            for stream in streams:
                stream.synchronize()

            for result in results:
                omega = float(result)
                omega_elements.append(omega)

        return zip(date_combinations, omega_elements)


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
                f"{self.data_source}_"
                f"{self.freq}"
            )
        else:
            self.omega_name = (
                f"omega_{self.kernel}_"
                f"{self.c}_"
                f"{self.ultimate_sample_start_date}_"
                f"{self.ultimate_sample_end_date}_"
                f"{self.characteristics_model}_"
                f"{self.data_source}_"
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

    def compute_omega(self,
        characteristics: pd.DataFrame,
        asset_rets: pd.Series
    ):
        """Main function of the class. It computes the omega matrix for CK-PCA."""

        # Get the unique dates in the sample
        sample_dates = asset_rets.index.get_level_values(1).unique().sort_values()

        # Pre-transfer all required data to GPU
        characteristics_device = {date: cp.asarray(characteristics.xs(date, level=1).to_numpy(), dtype=cp.float32) for date in sample_dates}
        asset_rets_device = {date: cp.asarray(asset_rets.xs(date, level=1).to_numpy(dtype=cp.float32)) for date in sample_dates}

        # Initialize omega as a DataFrame
        self.omega = pd.DataFrame(index=sample_dates, columns=sample_dates)

        # Create unique date combinations where d1 <= d2
        date_combinations = [(d1, d2) for i, d1 in enumerate(sample_dates) for d2 in sample_dates[i:]]

        # Parallel computation of omega
        omega_elements = self.process_date_combinations_cuda(characteristics_device, asset_rets_device, date_combinations)

        # Populate the omega DataFrame with the chunk results
        for (date1, date2), value in omega_elements:
            self.omega.loc[date1, date2] = value
            self.omega.loc[date2, date1] = value

        self.omega = self.omega.astype(float)

        # Cache omega to pickle file
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
        """_summary_

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

