###################################################
# Author: Ant Bilic                               #
# Since: Feb 22, 2024                             #
# Copyright: DAFF Hitchhikers working group       #
# Version: N/A                                    #
# Maintainer: Ante Bilic                          #
# Email: ante.bilic.mr@gmail.com                  #
# Status: N/A                                     #
###################################################

"""
This module contains source implementing how inspections are done along the pathway,
such as random sampling and CSP sampling
"""

from typing import NoReturn, override
from pathlib import Path
import sys
from dataclasses import dataclass, field
from importlib.resources import files
import itertools
import math
from functools import lru_cache
from copy import deepcopy
import numpy as np
from numpy.random import Generator
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from loguru import logger
from flatpack import data

logger.remove()
logger.add(sys.stderr, level="INFO")
type Vector[T] = np.ndarray[tuple[int], np.dtype[T]]
type ArrayLike[T] = Vector[T] | Series | DataFrame


def rn_random(size: int | tuple[int] | None = None) -> float | Vector[np.float_]:
    """
    Get a random float or an array of floats from the [0.0, 1.0) interval

    Parameters
    ----------
    size : Optional[int, Tuple[int]], optional
        The output shape. The default is None (i.e., return a single item)
    size : int
        The shape of the output array of random numbers to be generated.
        The default is None (only a single random float is generated).

    Returns
    -------
    ...: Union[float, Vector[float]]
        The rundom number, or an array of rundom numbers

    """
    rng: Generator = np.random.default_rng()
    return rng.random(size=size)


def get_column(df_in: DataFrame, col_name: str, fillval: bool | None = None) -> Series:
    """
    Select a column from the Dataframe if available, otherwise create it, fill it with the dessired
    value and return as Series.

    Parameters
    ----------
    df_in : DataFrame
        The dataframe with virtual containers processed (getting ready for output).
    col_name : str
        The name of the column of interest.
    fillval : Optional[bool], optional
        The value to impute if the column is missing. The default is None.

    Returns
    -------
    ... : Series
        The column of interest.

    """
    try:
        return df_in[col_name]
    except KeyError:
        df_in[col_name] = [fillval for _ in range(df_in.shape[0])]
        return df_in[col_name]


@dataclass(kw_only=True)
class SamplingRegime:
    """
    The top class an instance of which represents a particular container inspection regime.

    Parameters
    ----------
    sensitivity : float
        The probability of detecting contamination when present.

    """
    sensitivity: float
    contam_names: tuple[str, ...] = field(init=False, default=('LLC', 'HLC', 'Pest', 'yes'))
    inspection_count: int = field(init=False, default=0)
    contamination_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not isinstance(self.sensitivity, float | int):
            raise TypeError("sensitivity should be a number")
        if (self.sensitivity < 0) or (self.sensitivity > 1):
            raise ValueError("sensitivity value needs to be in the (0, 1) range")


    def inspect_container(self, container: dict) -> bool:
        """
        If contaminated and still undetected, detect contamination with the probability equal to the
        sensitivity value. It also keeps track of virtual container inspection & detection counts.

        Parameters
        ----------
        container : Dict
            A row from the DataFrame of containers with their attributes, representing a container.

        Returns
        -------
        ... : bool
            True if contaminated, previously undetected and then detected here. Otherwise False.

        """
        self.inspection_count += 1
        if container['contamination'] in self.contam_names:
            # checking that it hasn't already been detected and removed:
            if not container['detected']:
                if rn_random() < self.sensitivity:
                    self.contamination_count += 1
                    return True
        return False


    def inspect_batch(self, containers: DataFrame, dmask: ArrayLike[bool]) -> NoReturn:
        """
        Merely a placeholder, actual implementations inside the subclasses
        """
        raise NotImplementedError("Implementation varies based on the sampling protocol.")


@dataclass(kw_only=True)
class BasicSampler(SamplingRegime):
    """
    The class whose instance implements a sampling regime based simply on the monitoring fraction.

    Parameters
    ----------
    monit_frac : float
        monitoring fraction of containers on this sub-pathway

    """
    monit_frac: float

    def __post_init__(self) -> None:
        if not isinstance(self.monit_frac, float | int):
            raise TypeError("monit_frac should be a number")
        if (self.monit_frac < 0) or (self.monit_frac > 1):
            raise ValueError("monit_frac value needs to be in the (0, 1) range")


    @override
    def inspect_batch(self, containers: DataFrame, dmask: ArrayLike[bool]) -> None:
        """
        Inspection of virtual containers based on the sampling mask (dmask) and monitoring fraction

        Parameters
        ----------
        containers : DataFrame
            The virtual containers with their attributes.
        dmask : ArrayLike[bool]
            Decides which containers could be sampled (the True values).

        Returns
        -------
        None. Note, though, that the containers dataframe is updated with 3 new columns.

        """
        # Flag each container for monitoring with the probability of monit_frac:
        flags: Vector[bool] = rn_random(size=containers.shape[0]) < self.monit_frac
        # (new) boolean columns 'detected' and 'inspected', initially with False values:
        detected = get_column(containers, 'detected', fillval=False).copy()
        inspected = get_column(containers, 'inspected', fillval=False).copy()
        # Similarly, set up a new column with a list of directions for each row/container:
        if not 'directions' in containers.columns:
            containers['directions'] = [[] for _ in range(containers.shape[0])]
        # create a recursive deep copy (due to lists as values) of it in a separate Series:
        directions = pd.Series(deepcopy(containers['directions'].to_dict()))
        # Loop over the rows/containers selected for sampling via the mask:
        for cont in containers.loc[dmask, :].itertuples():
            # Select this container if flagged for monitoring (i.e., inspection):
            if flags[cont.Index]:
                directions[cont.Index].append('Container selected for monitoring')
                inspected[cont.Index] = True
                # If contaminated and still undetected, detect it with the probability=sensitivity:
                if self.inspect_container(cont._asdict()):
                    directions[cont.Index].append(f"{cont.contamination} found")
                    detected[cont.Index] = True
                # if nothing detected, release it:
                else:
                    directions[cont.Index].append('No contamination found, released')
            # Simply release it if not flagged:
            else:
                directions[cont.Index].append('Released')

        # Update the three columns for all containers:
        containers['directions'] = containers['directions'].mask(dmask, other=directions)
        containers.loc[:, 'detected'] = detected
        containers.loc[:, 'inspected'] = inspected


@lru_cache()
def load_ccv_lookup_table() -> DataFrame:
    """
    Read the table with recommended samples for CCV inspections and set a 4-level MultiIndex.

    Returns
    -------
    ... : DataFrame
        The table with recommended sampling values for Cargo Compliance Verification (CCV).

    """
    lookup_table = pd.read_csv(files(data).joinpath("recommended_samples.csv")).set_index(
        keys=['prior_samples', 'T1_level_percent', 'T2_percent', 'past_leakage']
        )
    return lookup_table


def set_prior_n_ccv(prior_n: int) -> int:
    """
    Get the value from 0-level MultIndex that is equal or just above the prior_n value
    (because the user provided prior_n wouldn't necessarily exist in the table).

    Parameters
    ----------
    prior_n : int
        The prior sample size

    Returns
    -------
    ... : int
        The updated prior sample size for CCV. It's the nearest (greater or equal) value
        to the prior_n parameter found in the table.

    """
    lookup_table = load_ccv_lookup_table()
    prior_n_vals = lookup_table.index.levels[0]
    # This returns the first closest value in prior_n_vals which is >= than prior_n:
    return next(itertools.dropwhile(lambda x: tuple(x)[1] < prior_n,
                                    zip(*itertools.tee(prior_n_vals))))[0]


def get_num_ccv_samples(prior_n: int,
                        prior_y: int,
                        t_risk: float,
                        t_change_level: int=95
                        ) -> int:
    """
    Either read or extrapolate the optimal sample size for the CCV inspections.

    Parameters
    ----------
    prior_n : int
        The prior sample size as the CCV parameter.
    prior_y : int
        The number of detected non-compliances.
    t_risk : float
        The acceptable threshold risk.
    t_change_level : int, optional
        The value of T1_level_percent from the CCV table, corresponding to T_change,
        i.e., the quantile of the rate of non-compliance distribution. The default is 95%.

    Raises
    ------
    ValueError
        Raised if no sample value can be extracted from the CCV table. This will typically
        occur when the risk (contamination rate) is too high.

    Returns
    -------
    ... : float
        Optimal sample size for CCV inspections, either read from the 'optimal_samples' column of
        the lookup_table or interpolated from there.

    """
    lookup_table = load_ccv_lookup_table()
    tmp = prior_n
    prior_n = set_prior_n_ccv(prior_n)
    if tmp != prior_n:
        logger.info(f"\nprior_n rounded-up from {tmp} to {prior_n}")
    # select the rows whose 3-level indices match the 3 parameters:
    try:
        relevant_rows = lookup_table.loc[(prior_n, t_change_level, t_risk), :]
    except KeyError as _e:
        logger.error(f"\nThe table does not include any matching row:\n{_e}")
        raise ValueError(
            f"No rows for prior_n={prior_n}, t_change_level={t_change_level}, t_risk={t_risk}"
        ) from _e

    try:
        # select the row whose past_leakage equals to prior_y:
        num_samples = relevant_rows.loc[prior_y, 'optimal_samples']
    except (KeyError, IndexError) as _e:
        # exact match of prior_y not the past_leakage values, look for an approximate:
        logger.exception(f"\nAttempting to extract num_samples from relevant_rows: {_e}")
        logger.info(f"\nrelevant_rows:\n{relevant_rows}")
        list_of_y = relevant_rows.index.to_list()
        y_below = 0
        y_above = 100_000
        found_below = False
        found_above = False
        for _y in list_of_y:
            if y_below < _y < prior_y:
                y_below = _y
                found_below = True
            if y_above > _y > prior_y:
                y_above = _y
                found_above = True
        if found_above and found_below:
            logger.info(f"\ny_above: {y_above}\ny_below: {y_below}")
            # interpolate between (y_below, optimal_samples) and (y_above, optimal_samples) points:
            num_samples_above = relevant_rows.loc[y_above, 'optimal_samples']
            num_samples_below = relevant_rows.loc[y_below, 'optimal_samples']
            slope = (num_samples_above - num_samples_below) / (y_above - y_below)
            intercept = num_samples_above - slope * y_above
            num_samples = slope * prior_y + intercept
            msg = f"""Resorted to interpolation for prior_n={prior_n},
            prior_y={prior_y},
            t_change_level={t_change_level}
            and t_risk={t_risk}.
            It produces num_samples: {num_samples}
            """
            logger.info(f"\n{msg}")
            return num_samples
        msg = f"""No value can be found for prior_n={prior_n},
        prior_y={prior_y},
        t_change_level={t_change_level}
        and t_risk={t_risk}.
        """
        raise ValueError(msg) from _e
    return num_samples


@dataclass(kw_only=True)
class RandomSampler(SamplingRegime):
    """
    The class whose instance implements the CCV sampling regime.

    Parameters
    ----------
    prior_n : int
        The prior sample size as the CCV parameter.
    prior_y : int
        The number of detected non-compliances.
    t_risk : float
        The acceptable threshold risk.
    t_change_level : int, optional
        The value of T1_level_percent from the CCV table, corresponding to T_change,
        i.e., the quantile of the rate of non-compliance distribution. The default is 95%

    """
    prior_n: int
    prior_y: int
    t_risk: float
    t_change_level: int=95

    def __post_init__(self) -> None:
        if not all(isinstance(_v, float | int) for _v in
                    (self.prior_n, self.prior_y, self.t_risk, self.t_change_level)):
            raise TypeError("The four CCV parameters must be positive numbers")
        if not all(_v >= 0 for _v in
                    (self.prior_n, self.prior_y, self.t_risk, self.t_change_level)):
            raise ValueError("The values must be POSITIVE")

        tmp = self.prior_n
        self.prior_n = set_prior_n_ccv(self.prior_n)
        if tmp != self.prior_n:
            logger.info(f"\nprior_n rounded-up from {tmp} to {self.prior_n}")


    @override
    def inspect_batch(self, containers: DataFrame, dmask: ArrayLike[bool]) -> None:
        """
        Inspection of virtual containers based on the sampling mask (dmask) and optimal sampling
        size based on the CCV paramaters.

        Parameters
        ----------
        containers : DataFrame
            The virtual containers with their attributes.
        dmask : ArrayLike[bool]
            Decides which containers could be sampled (the True values).

        Returns
        -------
        None. Note that the containers dataframe is updated with 3 new columns.

        """
        num_samples = get_num_ccv_samples(self.prior_n, self.prior_y, self.t_risk)
        # correct fraction of masked containers, make sure not > 1:
        sampling_fraction = min(1.0, num_samples / sum(dmask))
        contamination_found = 0
        samples_taken = 0
        # Flag each container for CCV sampling with the probability of sampling_fraction:
        flags: Vector[bool] = rn_random(size=containers.shape[0]) < sampling_fraction
        # (new) boolean columns 'detected' and 'inspected', initially with False values:
        detected = get_column(containers, 'detected', fillval=False).copy()
        inspected = get_column(containers, 'inspected', fillval=False).copy()
        # Similarly, set up a Series with a list of directions for each row/container:
        if not 'directions' in containers.columns:
            containers['directions'] = [[] for _ in range(containers.shape[0])]
        # create a recursive deep copy (due to lists as values) of it in a separate Series:
        directions = pd.Series(deepcopy(containers['directions'].to_dict()))
        # Loop over the rows/containers
        for cont in containers.loc[dmask, :].itertuples():
            # Select the container if flagged for CCV:
            if flags[cont.Index]:
                directions[cont.Index].append('Container selected for CCV')
                inspected[cont.Index] = True
                samples_taken += 1
                # If contaminated and still undetected, detect it with the probability=sensitivity:
                if self.inspect_container(cont._asdict()):
                    directions[cont.Index].append(f"{cont.contamination} found")
                    contamination_found += 1
                    detected[cont.Index] = True
                # or, if no contamaination found, release it:
                else:
                    directions[cont.Index].append('No contamination found, released')
            # Simply release it if not flagged:
            else:
                directions[cont.Index].append('Released')

        # Update the three columns for all containers:
        containers['directions'] = containers['directions'].mask(dmask, other=directions)
        containers.loc[:, 'detected'] = detected
        containers.loc[:, 'inspected'] = inspected
        # update prior_y using the newly found contaminations (i.e., contamination_found):
        tmp = self.prior_y
        self.prior_y = math.ceil(self.prior_y * (1 - samples_taken / self.prior_n)) + \
                        contamination_found
        if tmp != self.prior_y:
            logger.info(f"\nprior_y updated from {tmp} to {self.prior_y}")


@dataclass(kw_only=True)
class CSP1Sampler(SamplingRegime):
    """
    The class whose instance implements the CSP1 sampling regime.

    Parameters
    ----------
    clearance_num : int
        clearance number (CN) for the CSP1 sampling protocol..
    monit_frac : float
        monitoring fraction.

    """
    clearance_num: int
    monit_frac: float

    def __post_init__(self) -> None:
        if not all(isinstance(_v, float | int) for _v in
                    (self.clearance_num, self.monit_frac)):
            raise TypeError("The clearance_num and monit_frac must be numbers")
        if self.clearance_num <= 0:
            raise ValueError("clearance_num value must be POSITIVE")
        if (self.monit_frac < 0) or (self.monit_frac > 1):
            raise ValueError("monit_frac value needs to be in the (0, 1) range")


    @override
    def inspect_batch(self, containers: DataFrame, dmask: ArrayLike[bool]) -> None:
        """
        Inspection of virtual containers based on the sampling mask (dmask) and CSP1 parameters
        clearance number (CN) and monitoring fraction.

        Parameters
        ----------
        containers : DataFrame
            The virtual containers with their attributes.
        dmask : ArrayLike[bool]
            Decides which containers could be sampled (the True values).

        Returns
        -------
        None. Note that the containers dataframe is updated with 3 new columns.

        """
        # Either read values of all available importers or set as the value "default"
        if 'importer' in containers.columns:
            importers = containers['importer'].unique().tolist()
        else:
            importers = ['default']
            containers['importer'] = ["default" for _ in  range(containers.shape[0])]
        # Set the same clearance_num for each importer:
        census_remaining = {imp: self.clearance_num for imp in importers}

        # (new) boolean columns 'detected' and 'inspected', initially with False values:
        detected = get_column(containers, 'detected', fillval=False).copy()
        inspected = get_column(containers, 'inspected', fillval=False).copy()
        # Similarly, set up a Series with a list of directions for each row/container:
        if not 'directions' in containers.columns:
            containers['directions'] = [[] for _ in range(containers.shape[0])]
        # create a recursive deep copy (due to lists as values) of it in a separate Series:
        directions = pd.Series(deepcopy(containers['directions'].to_dict()))
        # Loop over the rows/containers
        for cont in containers.loc[dmask, :].itertuples():
            imp = cont.importer
            # Check the Importer's census status/mode:
            if census_remaining[imp] > 0:
                # The Importer subject to the census mode
                directions[cont.Index].append('Container inspected under census mode')
                inspected[cont.Index] = True
                # If contaminated and still undetected, detect it with the probability=sensitivity:
                if self.inspect_container(cont._asdict()):
                    directions[cont.Index].append(f"{cont.contamination} found")
                    census_remaining[imp] = self.clearance_num
                    detected[cont.Index] = True
                # Otherwise update Importer census_remaining and release the container
                else:
                    census_remaining[imp] -= 1
                    directions[cont.Index].append('No contamination found, released')
                    if census_remaining[imp] == 0:
                        directions[cont.Index].append(f"Importer {imp} removed from census mode")
            else:
                # The Importer is in the monitoring mode, select the container with prob=monit_frac
                if rn_random() < self.monit_frac:
                    directions[cont.Index].append('Container selected for monitoring')
                    inspected[cont.Index] = True
                    # If contaminated & undetected yet, detect it with the probability=sensitivity:
                    if self.inspect_container(cont._asdict()):
                        directions[cont.Index].append(f"{cont.contamination} found")
                        # Update Importer's census status/mode:
                        directions[cont.Index].append(f"Importer {imp} set to census mode")
                        census_remaining[imp] = self.clearance_num
                        detected[cont.Index] = True
                    else:
                        directions[cont.Index].append('No contamination found, released')
                # Simply release if not selected:
                else:
                    directions[cont.Index].append('Released')

        # Update the three columns for all containers:
        containers['directions'] = containers['directions'].mask(dmask, other=directions)
        containers.loc[:, 'detected'] = detected
        containers.loc[:, 'inspected'] = inspected


@dataclass(kw_only=True)
class CSP3Sampler(SamplingRegime):
    """
    The class whose instance implements the CSP3 sampling regime.

    Parameters
    ----------
    clearance_num : int
        clearance number (CN) for the CSP3 sampling protocol.
    monit_frac : float
        monitoring fraction.
    tight_cens_num: int
        tight census (TC) number for the CSP3 sampling protocol.

    """
    clearance_num: int
    monit_frac: float
    tight_cens_num: int

    def __post_init__(self) -> None:
        if not all(isinstance(_v, float | int) for _v in
                    (self.clearance_num, self.monit_frac, self.tight_cens_num)):
            raise TypeError("The clearance_num, monit_frac and tight_cens_num should be numbers")
        if not all(_v >= 0 for _v in (self.clearance_num, self.monit_frac, self.tight_cens_num)):
            raise ValueError("The three values must be POSITIVE")
        if self.monit_frac > 1:
            raise ValueError("monit_frac value needs to be in the (0, 1) range")


    @override
    def inspect_batch(self, containers: DataFrame, dmask: ArrayLike[bool]) -> None:
        """
        Inspection of virtual containers based on the sampling mask (dmask) and CSP3 parameters
        clearance number (CN), tight census number (TC) and monitoring fraction.

        Parameters
        ----------
        containers : DataFrame
            The virtual containers with their attributes.
        dmask : ArrayLike[bool]
            Decides which containers could be sampled (the True values).

        Returns
        -------
        None. Note that the containers dataframe is updated with 3 new columns.

        """
        if 'importer' in containers.columns:
            importers = containers['importer'].unique().tolist()
        else:
            importers = ['default']
            containers['importer'] = ['default' for _ in range(containers.shape[0])]
        # Set all Importers subject to "census" mode, starting with zero compliance_streak:
        modes = {imp: "census" for imp in importers}
        compliance_streak = {imp: 0 for imp in importers}

        # (new) boolean columns 'detected' and 'inspected', initially with False values:
        detected = get_column(containers, 'detected', fillval=False).copy()
        inspected = get_column(containers, 'inspected', fillval=False).copy()
        # Similarly, set up a Series with a list of directions for each row/container:
        if not 'directions' in containers.columns:
            containers['directions'] = [[] for _ in range(containers.shape[0])]
        # create a recursive deep copy (due to lists as values) of it in a separate Series:
        directions = pd.Series(deepcopy(containers['directions'].to_dict()))
        # Loop over the rows (i.e., the containers):
        for cont in containers.loc[dmask, :].itertuples():
            # The Importer and directions for this container:
            imp = cont.importer
            # For Importer subject to (tight)census carry out the required inspection:
            if modes[imp] == 'census' or modes[imp] == 'tight census':
                directions[cont.Index].append(f'Container inspected under {modes[imp]} mode')
                inspected[cont.Index] = True
                # If contaminated and still undetected, detect it with the probability=sensitivity:
                if self.inspect_container(cont._asdict()):
                    directions[cont.Index].append(f"{cont.contamination} found")
                    compliance_streak[imp] = 0
                    detected[cont.Index] = True
                    # Update Importer's census status/mode if appropriate:
                    if modes[imp] == 'tight census':
                        directions[cont.Index].append(f'Importer {imp} set to census mode')
                        modes[imp] = 'census'
                # Otherwise update Importer compliance_streak and container directions
                else:
                    compliance_streak[imp] += 1
                    directions[cont.Index].append('No contamination found, released')
                    # If meeting criteria, update, update Importer's census mode/status:
                    if modes[imp] == 'census' and compliance_streak[imp] >= self.clearance_num:
                        directions[cont.Index].append(f"Importer {imp} set to monitoring mode")
                        modes[imp] = 'monitoring'
                    elif modes[imp] == 'tight census' and \
                        compliance_streak[imp] >= self.tight_cens_num:
                        directions[cont.Index].append(
                            f"Importer {imp} set to failure detection mode")
                        modes[imp] = 'failure detection'
            # Otherwise inspect it with the probability=monit_frac:
            else:
                if rn_random() < self.monit_frac:
                    directions[cont.Index].append(f'Container inspected under {modes[imp]} mode')
                    inspected[cont.Index] = True
                    # If contaminated & undetected yet, detect it with the probability=sensitivity:
                    if self.inspect_container(cont._asdict()):
                        directions[cont.Index].append(f"{cont.contamination} found")
                        compliance_streak[imp] = 0
                        detected[cont.Index] = True
                        # If meeting criteria, update, update Importer's census mode/status:
                        if modes[imp] == 'monitoring':
                            directions[cont.Index].append(
                                f"Importer {imp} set to tight census mode")
                            modes[imp] = 'tight census'
                        elif modes[imp] == 'failure detection':
                            directions[cont.Index].append(f"Importer {imp} set to census mode")
                            modes[imp] = 'census'
                    #  Otherwise update Importer compliance_streak and container directions
                    else:
                        directions[cont.Index].append('No contamination found, released')
                        compliance_streak[imp] += 1
                        # If meeting criteria, update, update Importer's census mode/status:
                        if modes[imp] == 'failure detection' and \
                        compliance_streak[imp] >= self.clearance_num:
                            directions[cont.Index].append(f"Importer {imp} set to monitoring mode")
                            modes[imp] = 'monitoring'

        # Update the three columns for all containers:
        containers['directions'] = containers['directions'].mask(dmask, other=directions)
        containers.loc[:, 'detected'] = detected
        containers.loc[:, 'inspected'] = inspected


if __name__ == '__main__':
    """
    Below is a simple demo of this module for the external container inspections.
    The chosen sampling protocol (currently the CSP3) will be carried out on the dozen of virtual
    containers. If you wish to try a protocol, simply uncomment the two lines following the sampler
    set-up line (e.g., the two lines following basic_sampler = BasicSampler(...)).
    Note that calling each sampler's inspect_batch() method updates the dataframe of virtual
    containers, so if you wish to run 2, 3 or all 4 protocols, one must re-read the input CSV file
    after the dataframe gets exported as the containers_..._masked.csv file.
    """
    cs_df = pd.read_csv(Path.cwd() / "containers" / \
                        "original_ext_int_containers_external_with_medium.csv",
                        index_col=0)

    # Randomly set ~90% values to True, ~10% to False:
    the_mask = rn_random(size=cs_df.shape[0]) < 0.9

    basic_sampler = BasicSampler(sensitivity=0.9,  monit_frac=0.5)
    # basic_sampler.inspect_batch(cs_df, the_mask)
    # cs_df.to_csv(Path.cwd() / "containers_basic_masked.csv")

    random_sampler = RandomSampler(sensitivity=0.9, prior_n=4505, prior_y=0, t_risk=0.5)
    # random_sampler.inspect_batch(cs_df, the_mask)
    # cs_df.to_csv(Path.cwd() / "containers_random_masked.csv")

    csp1_sampler = CSP1Sampler(sensitivity=0.9, clearance_num=3, monit_frac=0.5)
    # csp1_sampler.inspect_batch(cs_df, the_mask)
    # cs_df.to_csv(Path.cwd() / "containers_csp1_masked.csv")

    csp3_sampler = CSP3Sampler(sensitivity=0.9, clearance_num=3, monit_frac=0.5, tight_cens_num=1)
    csp3_sampler.inspect_batch(cs_df, the_mask)
    cs_df.to_csv(Path.cwd() / "containers_csp3_masked.csv")
