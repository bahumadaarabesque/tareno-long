import sys

from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np

from sray_db.apps import app, apps
from sray_db.apps.field import Field
from sray_db.apps.pk import PrimaryKey
from sray_db.broker import DataBroker
from sray_db.query import Get, Put
from structlog import get_logger

logger = get_logger()


class SRayDBLoader:
    def load(self, df: pd.DataFrame, load_cutoff: datetime = None,
             merge_index: np.ndarray = np.ndarray(0), specific_assets: Optional[List[int]] = None) -> None:

        if merge_index.any():
            df = df.loc[merge_index]

        load_idx_to_overwrite = self.get_load_idx_to_overwrite(df, specific_assets)

        try:
            logger.info("Attempting to push %d rows to the database...", len(df))
            db = DataBroker()
            put_query = Put(df, load_cutoff, load_idx_to_overwrite)
            rows_updated = db.query(put_query)
        except Exception as e:
            logger.exception("Failed to push data to database", error=e)
            sys.exit(1)
        else:
            logger.info(
                "Successfully pushed %d rows to the database", rows_updated)

    def cols_to_fields(self, app: app.App, fields: List[str]) -> List[Field]:
        """
        Converts column names to sray_db fields
        """
        return [app[col] for col in fields]

    def fields_to_cols(self, fields: List[Field]) -> List[str]:
        """
        Converts sray_db fields to str
        """
        columns = []

        for col in fields:
            if col.name == 'report_date' and col.app.name == 'wsje_tr_scores':
                columns.append('tr_report_date')
            else:
                columns.append(col.name)

        return columns

    def fetch(self, fields: List[Field], query_assets: pd.Index,
              query_dates: pd.DatetimeIndex = pd.DatetimeIndex([]),
              as_of: str = None) -> pd.DataFrame:
        db = DataBroker()

        if not query_dates.empty and not query_assets.empty:
            get_query = Get(
                fields, load_idx=[query_assets, query_dates], as_of=as_of)
        elif query_dates.empty and not query_assets.empty:
            get_query = Get(
                fields, load_idx=[query_assets], as_of=as_of)
        else:
            get_query = Get(fields, load_idx=[query_dates], as_of=as_of)

        df = db.query(get_query)

        cols_to_drop = [col for col in fields if col.name in ['load_cutoff', 'date_created']]

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df

    @staticmethod
    def get_load_idx_to_overwrite(df: pd.DataFrame, specific_assets: Optional[List[int]] = None) \
            -> Optional[pd.Index]:

        load_idx_to_overwrite = None
        if not specific_assets:
            load_idx_to_overwrite = [df.index.get_level_values(PrimaryKey.date).unique()]

        return load_idx_to_overwrite
