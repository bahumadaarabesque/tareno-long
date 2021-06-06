from collections import defaultdict
from typing import Dict, List

import pandas as pd


class FeatureLongDimension:
    def __init__(self, features: List[str]) -> None:
        self.dimension_feature_mapping = self.create_dimension_feature_mapping(features)

    def create_dimension_feature_mapping(self, features: List[str]) -> Dict[str, List[str]]:
        """
        Creates a mapping between SASB Standard Dimensions and their respective SASB GICs/features
        Each GIC has the format <dimension>_<category>, so we split by the '_' and the first
        item is the dimension

        Args:
            features (List[str]): List of features

        Returns:
            Dict[str, List[str]]: A dictionary mapping Dimensions to GICs
        """
        dimension_feature_mapping = defaultdict(list)
        for feature in features:
            dimension = feature.split('_')[0]
            dimension_feature_mapping[dimension].append(feature)

        return dimension_feature_mapping

    def calc(self, feature_long_scores: pd.DataFrame, adj: bool) -> pd.DataFrame:
        """
        If adj == False: Compute SD scores, from the mean of the GICs/features that relate to them
        Othwerwise compute SD scores, from the sum

        Args:
            feature_long_scores (pd.DataFrame): Long scores. Either adjusted or unadjusted

        Returns:
            pd.DataFrame: A dataframe with SD scores
        """
        dimensions = list(self.dimension_feature_mapping.keys())
        dimension_df = pd.DataFrame(index=feature_long_scores.index, columns=dimensions)

        for dimension in dimensions:
            dimension_features = self.dimension_feature_mapping[dimension]

            if adj:
                dimension_df[dimension] = feature_long_scores[dimension_features].sum(
                    axis=1)
            else:
                dimension_df[dimension] = feature_long_scores[dimension_features].mean(
                    axis=1)

        return dimension_df
