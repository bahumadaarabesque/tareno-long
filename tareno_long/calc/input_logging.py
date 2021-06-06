import pandas as pd
import numpy as np

from typing import Dict, List, Tuple
from sray_db.apps.pk import PrimaryKey


class InputLogging:
    fetures: List[str]
    feature_input_mapping: Dict[str, List[str]]
    industry_materiality_dict: Dict[str, Dict[str, List[str]]]
    input_logging_columns: List[str]

    def __init__(
        self,
        features: List[str],
        feature_input_mapping: Dict[str, List[str]],
        industry_materiality_dict: Dict[str, Dict[str, List[str]]],
        input_logging_columns: List[str],
    ) -> None:
        self.features = features
        self.feature_input_mapping = feature_input_mapping
        self.industry_materiality_dict = industry_materiality_dict
        self.input_logging_columns = input_logging_columns

    def calc(self, feature_input_df: pd.DataFrame) -> pd.DataFrame:
        """
        This function does the following:
        - Indexes by industry
        - Groups features by their inputs
        - Counts number of inputs per feature which are not nan
        - Counts total possible material features and assigns this as available_material_features
        - Counts total possible inputs per material feature and assigns this as available_material_inputs
        - Counts live material features (>= 1 input per material feature)
        - Counts live immaterial features (>= 1 input per immaterial feature)
        - Counts active material features (>= 2 inputs per material feature)

        Args:
            feature_input_df (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """
        result_dfs = []
        feature_input_df = feature_input_df.reset_index()
        merged = feature_input_df.set_index(
            ["sasb_industry", PrimaryKey.assetid, PrimaryKey.date]
        )

        self.instantiate_cols(merged)

        for industry in self.industry_materiality_dict:
            if not merged.index.isin([industry], level=0).any():
                continue

            feature_types = self.industry_materiality_dict[industry]
            material_features = feature_types["material"]
            immaterial_features = feature_types["immaterial"]
            available_material_inputs = sum(
                [
                    len(self.feature_input_mapping[feature])
                    for feature in material_features
                ]
            )

            industry_df = merged.loc[industry, :]
            industry_df["sasb_industry"] = industry

            industry_df.loc[
                industry_df.index.values, "available_material_features"
            ] = len(material_features)

            industry_df.loc[
                industry_df.index.values, "available_material_inputs"
            ] = available_material_inputs

            for feature in self.features:
                feature_inputs = self.feature_input_mapping[feature]
                feature_input_scores = industry_df[feature_inputs]

                non_nan_feature_inputs = feature_input_scores.count(axis=1)
                industry_df.loc[
                    industry_df.index.values, feature + "_input_counts"
                ] = non_nan_feature_inputs

            material_feature_input_counts = industry_df[
                [feature + "_input_counts" for feature in material_features]
            ]
            immaterial_feature_input_counts = industry_df[
                [feature + "_input_counts" for feature in immaterial_features]
            ]

            (
                live_material_features,
                material_features_atleast_2_inputs,
            ) = self.get_live_and_active_features(material_feature_input_counts)

            (
                live_material_feature_count,
                live_immaterial_feature_count,
                material_features_atleast_2_inputs_count,
            ) = self.get_live_and_active_counts(
                live_material_features,
                immaterial_feature_input_counts,
                material_features_atleast_2_inputs,
            )

            if not live_material_features.isnull().all().all():
                self.populate_material_features_names(
                    live_material_features, industry_df
                )

            self.populate_live_and_active_features(
                live_material_feature_count,
                industry_df,
                live_immaterial_feature_count,
                material_features_atleast_2_inputs_count,
            )

            material_inputs_used = material_feature_input_counts.sum(axis=1)
            industry_df.loc[
                live_material_feature_count.index.values, "material_inputs_used"
            ] = material_inputs_used

            self.populate_use_ratio(industry_df, live_material_feature_count)

            if not material_features_atleast_2_inputs.isnull().all().all():
                self.populate_material_features_names(
                    material_features_atleast_2_inputs,
                    industry_df,
                    "material_features_with_at_least_2_inputs_names",
                )

            result_dfs.append(industry_df)

        input_logging_df = pd.concat(result_dfs)
        input_logging_df = input_logging_df[
            self.input_logging_columns + ["live_immaterial_features", "sasb_industry"]
        ]
        input_logging_df = input_logging_df.dropna(how="all")

        return input_logging_df

    def get_live_and_active_features(
        self, material_feature_input_counts: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        live_material_features = material_feature_input_counts[
            material_feature_input_counts >= 1
        ]
        material_features_atleast_2_inputs = material_feature_input_counts[
            material_feature_input_counts >= 2
        ]
        return live_material_features, material_features_atleast_2_inputs

    def populate_live_and_active_features(
        self,
        live_material_feature_count: pd.Series,
        industry_df: pd.DataFrame,
        live_immaterial_feature_count: pd.Series,
        material_features_atleast_2_inputs_count: pd.Series,
    ) -> None:
        industry_df.loc[
            industry_df.index.values, "live_material_features"
        ] = live_material_feature_count

        industry_df.loc[
            industry_df.index.values, "live_immaterial_features"
        ] = live_immaterial_feature_count

        industry_df.loc[
            industry_df.index.values, "material_features_with_at_least_2_inputs"
        ] = material_features_atleast_2_inputs_count

    def get_live_and_active_counts(
        self,
        live_material_features: pd.DataFrame,
        immaterial_feature_input_counts: pd.DataFrame,
        material_features_atleast_2_inputs: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        live_material_feature_count = live_material_features.count(axis=1)
        live_immaterial_feature_count = immaterial_feature_input_counts[
            immaterial_feature_input_counts >= 1
        ].count(axis=1)

        material_features_atleast_2_inputs_count = (
            material_features_atleast_2_inputs.count(axis=1)
        )
        return (
            live_material_feature_count,
            live_immaterial_feature_count,
            material_features_atleast_2_inputs_count,
        )

    def populate_use_ratio(
        self, industry_df: pd.DataFrame, live_material_feature_count: pd.Series
    ) -> None:
        use_ratio = (
            industry_df.loc[
                live_material_feature_count.index.values, "material_inputs_used"
            ]
            / industry_df.loc[
                live_material_feature_count.index.values,
                "available_material_inputs",
            ]
        )
        industry_df.loc[
            live_material_feature_count.index.values,
            "use_ratio",
        ] = use_ratio

    def populate_material_features_names(
        self,
        live_material_features: pd.DataFrame,
        industry_df: pd.DataFrame,
        col_to_populate: str = "material_features_names",
    ) -> None:
        """
        Does some dot product magic to find the column names where there are non nan entries
        This is then made into a series and is added to the dataframe as col_to_populate

        Args:
            live_material_features (pd.DataFrame): live or active material features df. nan entries are for what are not either
            industry_df (pd.DataFrame): dataframe representing industry
            col_to_populate (str, optional): name of column to put names into. Defaults to "material_features_names".
        """

        live_material_features_names = (
            live_material_features.notna()
            .dot(live_material_features.columns + ",")
            .str.rstrip(",")
        )

        live_material_features_names = live_material_features_names.apply(
            lambda x: x.replace("_input_counts", "")
        )
        industry_df.loc[
            live_material_features.index.values,
            col_to_populate,
        ] = live_material_features_names

    def instantiate_cols(self, merged: pd.DataFrame) -> None:
        for feature in self.features:
            merged[feature + "_input_counts"] = np.nan

        for input_column in self.input_logging_columns:
            if input_column == 'tr_report_date':
                continue
            merged[input_column] = np.nan

        for industry in self.industry_materiality_dict:
            merged[industry] = np.nan

        merged["material_features_names"] = np.nan
        merged["material_features_with_at_least_2_inputs"] = np.nan
        merged["material_features_with_at_least_2_inputs_names"] = np.nan
