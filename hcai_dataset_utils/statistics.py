import pandas as pd
from abc import ABC, abstractmethod


class Statistics(ABC):
    @property
    @abstractmethod
    def info(self):
        ...

    def _populate_stats(self, data, split="train"):
        if isinstance(data, pd.DataFrame):
            stats = self.stats_from_dataframe(data)
            self.info.metadata[split] = stats
        else:
            raise TypeError()

    @staticmethod
    def stats_from_dataframe(df: pd.DataFrame):
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
