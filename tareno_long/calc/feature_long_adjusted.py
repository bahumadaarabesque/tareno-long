import pandas as pd


class FeatureLongScoringAdjusted:
    def calc(
        self, feature_long_scores: pd.DataFrame, materiality_weights: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates the adjusted feature long score

            1. The materiality weights df is combined with the
               `feature_long_score` df to make them the same size to
               carry out operations on
            2. It is then reindexed like `feature_long_score`
        """

        adjusted_long_scores = (feature_long_scores - 50) * materiality_weights

        return adjusted_long_scores
