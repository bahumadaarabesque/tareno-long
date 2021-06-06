from typing import Dict, List

import pandas as pd
from sray_db.apps.pk import PrimaryKey

MATERIAL_WEIGHT = 0.8
IMMATERIAL_WEIGHT = 0.2


class MaterialityWeighting:
    """
    Applies materiality weighting to features based on
     MATERIAL_WEIGHT and IMMATERIAL_WEIGHT
    """

    features: List[str]
    industry_materiality_dict: Dict[str, List[str]]

    def __init__(self, features: List[str], industry_materiality_dict: Dict[str, List[str]]) -> None:
        self.features = features
        self.industry_materiality_dict = industry_materiality_dict

    def calc(self, input_logging_df: pd.DataFrame) -> pd.DataFrame:
        """
        On an industry basis, get the material and immaterial features and
         apply a weighting
        """
        weight_df = pd.DataFrame(
            index=input_logging_df.index, columns=self.features
        )

        input_logging_df = input_logging_df.reset_index().set_index(
            ['sasb_industry', PrimaryKey.assetid, PrimaryKey.date])

        for industry in self.industry_materiality_dict:
            if not input_logging_df.index.isin([industry], level=0).any():
                continue

            industry_input_logging_df = input_logging_df.loc[industry, :]
            material_features = self.industry_materiality_dict[industry]['material']
            immaterial_features = self.industry_materiality_dict[industry]['immaterial']

            live_material_feature_df = industry_input_logging_df['live_material_features']
            live_immaterial_feature_df = industry_input_logging_df['live_immaterial_features']

            live_material_feature_weights = MATERIAL_WEIGHT / live_material_feature_df
            live_immaterial_feature_weights = IMMATERIAL_WEIGHT / live_immaterial_feature_df

            weight_df.loc[
                industry_input_logging_df.index.values, material_features
            ] = live_material_feature_weights

            weight_df.loc[
                industry_input_logging_df.index.values, immaterial_features
            ] = live_immaterial_feature_weights

        return weight_df
