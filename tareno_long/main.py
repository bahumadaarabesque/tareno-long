import sys
from dotenv import load_dotenv
load_dotenv()

from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from arabesque_py.logging.logging import setup

setup()

from arabesque_py.prometheus.prometheus import Prometheus
from arabesque_py.score_validation.long_total_validator import LongTotalValidator
from dateutil.parser import parse
from sray_db.apps import apps
from sray_db.apps.app import App
from sray_db.apps.field import Field
from sray_db.apps.pk import PrimaryKey
from structlog import get_logger

from tareno_long.arguments import args
from tareno_long.calc.feature_long import FeatureLongScoring
from tareno_long.calc.feature_long_adjusted import FeatureLongScoringAdjusted
from tareno_long.calc.feature_long_sd import FeatureLongDimension
from tareno_long.calc.input_logging import InputLogging
from tareno_long.calc.weights import MaterialityWeighting
from tareno_long.config.config import LongConfig
from tareno_long.industry_materiality_dict import industry_materiality_dict
from tareno_long.sray_db_load.loader import SRayDBLoader

pd.options.mode.chained_assignment = None

logger = get_logger()


def main(
    start_date: str,
    end_date: str,
    load_cutoff: str,
    specific_assets: list,
    validation_override: bool,
) -> None:
    """
    Calculates unadjusted feature mean
        - loads SASB materiality data
        - generates an input mapping from input_name to GIC
    """
    config = LongConfig()
    gateway = gateway = f"{config.gateway_host}:{config.gateway_port}"
    prometheus = Prometheus(threaded=False, gateway=gateway)
    validator = LongTotalValidator("wsje_long", prometheus, lower_limit=-50, upper_limit=50)
    try:
        calculate_and_load_scores(
            start_date,
            end_date,
            load_cutoff,
            specific_assets,
            validation_override,
            validator,
        )
    except Exception as e:
        logger.exception("Something went wrong in the calculation", error=e)
        validator.set_app_exception()
        sys.exit(1)


def calculate_and_load_scores(
        start_date, end_date, load_cutoff, specific_assets, validation_override, validator
):
    wsje_tr_scores_app = apps["wsje_tr_scores"][(1, 1, 0, 0)]
    wsje_ir_scores_app = apps["wsje_ir_scores"][(1, 1, 0, 0)]
    sasb_sics_materiality_app = apps["sasb_sics_materiality"][(1, 0, 0, 0)]
    input_logging_app = apps["wsje_input_logging"][(1, 1, 0, 0)]
    input_logging_binary_app = apps["wsje_input_logging_binary"][(1, 1, 0, 0)]
    feature_long_app = apps["wsje_long"][(1, 1, 0, 0)]
    feature_long_un_adj_app = apps["wsje_long_un_adj"][(1, 1, 0, 0)]
    feature_long_sd_app = apps["wsje_long_sd"][(1, 1, 0, 0)]
    feature_long_sd_un_adj_app = apps["wsje_long_sd_un_adj"][(1, 1, 0, 0)]

    db = SRayDBLoader()

    feature_input_df = pd.read_csv(
        "./assets/input_metadata.csv", usecols=["gic", "input"]
    )
    ir_input_data = feature_input_df[feature_input_df["input"].str.startswith("ir")][
        "input"
    ].values.tolist()
    tr_input_data = feature_input_df[feature_input_df["input"].str.startswith("tr")][
        "input"
    ].values.tolist()
    tr_input_data.append('report_date')

    ir_fields = db.cols_to_fields(wsje_ir_scores_app, ir_input_data)
    tr_fields = db.cols_to_fields(wsje_tr_scores_app, tr_input_data)
    sasb_fields = [
        field
        for field in list(sasb_sics_materiality_app.values())
        if field.name not in ["load_cutoff", "date_created"]
    ]

    sasb_app_cols = db.fields_to_cols(sasb_fields)
    combined_fields = ir_fields + tr_fields + sasb_fields
    combined_input_cols = ir_input_data + tr_input_data
    query_dates = pd.date_range(start_date, end_date, freq="D", name=PrimaryKey.date)
    query_assets = (
        pd.Index([])
        if not specific_assets
        else pd.Index(specific_assets, name=PrimaryKey.assetid)
    )
    input_scores, input_logging_binary_df = calc_input_logging_binary(
        db,
        combined_fields,
        combined_input_cols,
        query_assets,
        query_dates,
        sasb_app_cols,
    )

    features = feature_input_df["gic"].unique().tolist()
    feature_input_mapping = _feature_input_mapping(feature_input_df)
    input_logging_df = calc_input_logging(
        features,
        feature_input_mapping,
        industry_materiality_dict,
        input_scores,
        input_logging_app,
    )
    feature_long_score = calculate_feat_long(
        feature_input_mapping, features, input_scores
    )
    weights = calculate_materiality_weights(
        features, input_logging_df, industry_materiality_dict
    )
    adjusted_long_scores = calculate_feat_long_adjusted(feature_long_score, weights)
    feature_long_sd_un_adj, feature_long_sd = calculate_feat_long_dimensions(
        features, feature_long_score, adjusted_long_scores
    )
    input_logging_survived_assets = input_logging_df[
        input_logging_df["material_features_with_at_least_2_inputs"] >= 2
    ].index.values

    if not validation_override:
        current_long_score = adjusted_long_scores.loc[input_logging_survived_assets]
        prev_start_date = pd.Timestamp(start_date) - timedelta(days=14)
        prev_end_date = pd.Timestamp(start_date) - timedelta(days=1)
        prev_date_range = pd.date_range(
            prev_start_date, prev_end_date, freq="D", name=PrimaryKey.date
        )
        previous_long_score = db.fetch(
            list(feature_long_app.values()), query_assets, prev_date_range, None
        )
        previous_long_score.columns = db.fields_to_cols(previous_long_score.columns)
        valid = validator.validate(previous_long_score, current_long_score)
        if not valid:
            logger.exception("Validation failed, exiting with code 1")
            sys.exit(1)

    feature_long_score.columns = db.cols_to_fields(
        feature_long_un_adj_app, feature_long_score.columns
    )
    adjusted_long_scores.columns = db.cols_to_fields(
        feature_long_app, adjusted_long_scores.columns
    )
    feature_long_sd_un_adj.columns = db.cols_to_fields(
        feature_long_sd_un_adj_app, feature_long_sd_un_adj.columns
    )
    feature_long_sd.columns = db.cols_to_fields(
        feature_long_sd_app, feature_long_sd.columns
    )
    binary_fields = db.cols_to_fields(
        input_logging_binary_app, list(input_logging_binary_df.columns)
    )
    input_logging_binary_df.columns = binary_fields
    input_logging_df.drop(
        ["live_immaterial_features", "sasb_industry"], axis=1, inplace=True
    )
    input_logging_df.columns = db.cols_to_fields(
        input_logging_app, input_logging_df.columns
    )

    logger.info("Loading input_logging...")
    db.load(input_logging_df, load_cutoff, merge_index=input_logging_survived_assets, specific_assets=specific_assets)
    logger.info("Loading input_logging_binary ...")
    db.load(
        input_logging_binary_df, load_cutoff, merge_index=input_logging_survived_assets, specific_assets=specific_assets
    )
    logger.info("Loading feature long scores...")
    db.load(feature_long_score, load_cutoff, merge_index=input_logging_survived_assets, specific_assets=specific_assets)
    logger.info("Loading feature long adjusted scores...")
    db.load(
        adjusted_long_scores, load_cutoff, merge_index=input_logging_survived_assets, specific_assets=specific_assets
    )
    logger.info("Loading feature long sd scores...")
    db.load(
        feature_long_sd_un_adj, load_cutoff, merge_index=input_logging_survived_assets, specific_assets=specific_assets
    )
    logger.info("Loading feature long sd adjusted scores...")
    db.load(feature_long_sd, load_cutoff, merge_index=input_logging_survived_assets, specific_assets=specific_assets)


def calculate_feat_long_dimensions(
    features: List[str],
    feature_long_score: pd.DataFrame,
    adjusted_long_scores: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Calculating dimension level scores")

    dimension_calc = FeatureLongDimension(features)
    feature_long_sd_un_adj = dimension_calc.calc(feature_long_score, adj=False)
    feature_long_sd = dimension_calc.calc(adjusted_long_scores, adj=True)

    logger.info("Completed calculating dimension level scores")
    return feature_long_sd_un_adj, feature_long_sd


def calculate_feat_long_adjusted(
    feature_long_score: pd.DataFrame, weights: pd.DataFrame
) -> pd.DataFrame:
    logger.info("Calculating feature long adjusted")
    feauture_long_scoring_adjusted = FeatureLongScoringAdjusted()
    adjusted_long_scores = feauture_long_scoring_adjusted.calc(
        feature_long_score, weights
    )

    logger.info("Completed feature long adjusted calculation")
    return adjusted_long_scores


def calculate_materiality_weights(
    features: List[str], input_logging_df: pd.DataFrame, industry_materiality_dict: Dict
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    logger.info("Calculating materiality weights")

    weighting_calc = MaterialityWeighting(features, industry_materiality_dict)
    weights = weighting_calc.calc(input_logging_df)

    logger.info("Materiality weights calculation complete")
    return weights


def calculate_feat_long(
    feature_input_mapping: Dict[str, List[str]],
    features: List[str],
    input_scores: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Calculating feature long")
    feature_long_calc = FeatureLongScoring(feature_input_mapping, features)
    feature_long_score = feature_long_calc.calc(input_scores)

    logger.info("Feature long calculation complete")
    return feature_long_score


def calc_input_logging_binary(
    db,
    combined_fields: List[Field],
    combined_input_cols: List[str],
    query_assets: pd.Index,
    query_dates: pd.Index,
    sasb_app_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Fetching input scores for %d", len(query_assets))

    input_scores = db.fetch(combined_fields, query_assets, query_dates, None)
    if input_scores.empty:
        logger.info("Input scores for %s are empty! Exiting early!")
        sys.exit(0)

    logger.info("Fetched %d rows for input scores", len(input_scores))

    input_scores.columns = db.fields_to_cols(input_scores.columns)
    input_scores = _remove_nan_sasb_rows(input_scores, sasb_app_cols)

    combined_input_cols.remove('report_date')
    input_logging_binary_df = input_scores[combined_input_cols].notnull().astype(int)
    input_logging_binary_df = input_logging_binary_df.join(input_scores['tr_report_date'])
    input_logging_binary_df = _fill_report_date(input_logging_binary_df)

    return input_scores, input_logging_binary_df


def calc_input_logging(
    features: List[str],
    feature_input_mapping: Dict[str, List[str]],
    industry_materiality_dict: Dict[str, List[str]],
    input_scores: pd.DataFrame,
    input_logging_app: App,
) -> pd.DataFrame:
    logger.info("Calculating input_logging")
    input_logging_columns = [col.name for col in list(input_logging_app.values())][3:]

    input_logging = InputLogging(
        features,
        feature_input_mapping,
        industry_materiality_dict,
        input_logging_columns,
    )

    input_logging_df = input_logging.calc(input_scores)
    input_logging_df = _fill_report_date(input_logging_df)

    logger.info("input_logging calculation complete")

    return input_logging_df


def _fill_report_date(df: pd.DataFrame) -> pd.DataFrame:
    df[["ir_report_date"]] = np.nan
    return df


def _remove_nan_sasb_rows(
    input_score_df: pd.DataFrame, sasb_column_names: List[str]
) -> pd.DataFrame:
    return input_score_df.dropna(axis=0, how="all", subset=sasb_column_names)


def _feature_input_mapping(feature_input_df: pd.DataFrame) -> Dict[str, List[str]]:
    feature_input_combinations = feature_input_df.set_index(
        ["gic", "input"]
    ).index.tolist()
    feature_input_mapping = defaultdict(list)

    for gic in feature_input_combinations:
        feature_input_mapping[gic[0]].append(gic[1])
    return feature_input_mapping


if __name__ == "__main__":
    start_date = args.start_date
    end_date = args.end_date
    load_cutoff = args.load_cutoff
    specific_assets = args.specific_assets
    validation_override = args.validation_override

    if specific_assets:
        specific_assets = list(map(int, specific_assets.split(",")))

    load_cutoff = parse(load_cutoff)

    main(start_date, end_date, load_cutoff, specific_assets, validation_override)
