import pandas as pd
from abc import ABC, abstractmethod


class Statistics(ABC):
    @property
    @abstractmethod
    def info(self):
        ...

    def _populate_stats(self, data: pd.DataFrame, split="train"):
        """
        Adding statistic metadata to the dataset information. Needs to manually called in the dataset child class.
        Args:
            data: The input data to analyze. Could be of type: pandas.DataFrame, ...
            split: The splitname that is set in the dictionary.
        """
        if isinstance(data, pd.DataFrame):
            stats = self.stats_from_dataframe(data)
        else:
            raise TypeError()
        self.info.metadata[split] = stats

    @staticmethod
    def stats_from_dataframe(df: pd.DataFrame) -> dict:
        """
        Calculates statistics for the given dataframe.
        The exact form of the return value depens on the input. The function will generate individual stats for each column in the dataframe.
        If the column is of dtype objects the calculated attributes are: count, uniqe, top, freq, dist.
        Else the statistic comprises the attributes: count, mean, std, min, max, percentiles 25%, 50%,75%.
        See pandas.dataframe.describe() for more information

        Esample output:
        {
        'expression':
            {'count': 382670,
            'unique': 11,
            'top': 1,
            'freq': 125245,
            'dist': {'1': 125245, '10': 75505, '0': 69841, '8': 30753, '2': 22690, '6': 22381, '3': 13077, '9': 10722, '4': 5581, '7': 3467, '5': 3408}
            },
        'arousal': {'count': 296443.0, 'mean': 0.08697717502906124, 'std': 0.32627918341864814, 'min': -0.998677, '25%': -0.085034, '50%': 0.0555556, '75%': 0.216057, 'max': 0.996903},
        'valence': {'count': 296443.0, 'mean': 0.1872163682477575, 'std': 0.5039924392122456, 'min': -0.999081, '25%': -0.161064, '50%': 0.203252, '75%': 0.626984, 'max': 0.999081}
        }

        Args:
            df: Input dataframe

        Returns:
            dict: Dictionary containing the calculated statistics

        """
        ret = {}

        # Categorical stats
        df_cat = df.select_dtypes(include=["object"])
        if not df_cat.empty:
            dsc_cat = df_cat.describe().to_dict()
            distribution = {
                l: {"dist": df_cat[l].value_counts(sort=True).to_dict()}
                for l in df_cat.head()
            }
            for k, v in distribution.items():
                dsc_cat[k].update(v)
            ret.update(dsc_cat)

        # Continuous stats
        df_cont = df.select_dtypes(exclude=["object"])
        if not df_cont.empty:
            dsc_cont = df_cont.describe().to_dict()
            ret.update(dsc_cont)

        return ret
