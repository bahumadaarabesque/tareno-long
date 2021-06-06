import sys
from dotenv import load_dotenv
load_dotenv()

from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sray_db.apps import apps
from sray_db.apps.app import App
from sray_db.apps.field import Field
from sray_db.apps.pk import PrimaryKey
from tareno_long.sray_db_load.loader import SRayDBLoader

wsje_tr_scores_app = apps["wsje_tr_scores"][(1, 1, 0, 0)]
wsje_ir_scores_app = apps["wsje_ir_scores"][(1, 1, 0, 0)]

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