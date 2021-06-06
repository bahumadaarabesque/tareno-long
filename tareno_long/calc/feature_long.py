import pandas as pd

from typing import Dict, List
from collections import defaultdict


class FeatureLongScoring:
    features: List[str]
    feature_input_mapping: Dict[str, List[str]]

    def __init__(self, feature_input_mapping: Dict[str, List[str]], features: List[str]) -> None:
        self.feature_input_mapping = feature_input_mapping
        self.features = features

    def calc(self, input_score_df: pd.DataFrame) -> pd.DataFrame:

        results_df = pd.DataFrame(
            index=input_score_df.index, columns=self.features)

        for feature in self.features:
            feature_inputs = self.feature_input_mapping[feature]
            results_df[feature] = input_score_df[feature_inputs].mean(axis=1)

        return results_df
