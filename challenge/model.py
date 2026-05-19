from datetime import datetime
import pandas as pd
import xgboost as xgb

from typing import Tuple, Union, List

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

        # Define time ranges for morning, afternoon and evening.
        self.morning_min = datetime.strptime("05:00", '%H:%M').time()
        self.afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        self.evening_min = datetime.strptime("19:00", '%H:%M').time()

        # Define date ranges for high season
        self.range1_min = datetime.strptime('15-Dec', '%d-%b')
        self.range1_max = datetime.strptime('31-Dec', '%d-%b')
        self.range2_min = datetime.strptime('1-Jan', '%d-%b')
        self.range2_max = datetime.strptime('3-Mar', '%d-%b')
        self.range3_min = datetime.strptime('15-Jul', '%d-%b')
        self.range3_max = datetime.strptime('31-Jul', '%d-%b')
        self.range4_min = datetime.strptime('11-Sep', '%d-%b')
        self.range4_max = datetime.strptime('30-Sep', '%d-%b')

        # Define the expected features
        self._EXPECTED_FEATURES = [
            'OPERA', 
            'MES', 
            'TIPOVUELO', 
            # 'SIGLADES', 
            # 'DIANOM'
        ]

        # Define the features to be used in the model.
        self._TOP_10_FEATURES = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]

    def _is_high_season(self, fecha: str) -> int:
        """
        Check if the date is in high season or not. High season is defined as the following date ranges:
        - 15-Dec to 3-Mar
        - 15-Jul to 31-Jul
        - 11-Sep to 30-Sep
        Args:
            fecha (str): date in the format 'YYYY-MM-DD HH:MM:SS'
        Returns:
            int: 1 if the date is in high season, 0 otherwise.
        """
        fecha_año = int(fecha.split('-')[0])
        fecha_dt = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = self.range1_min.replace(year = fecha_año)
        range1_max = self.range1_max.replace(year = fecha_año)
        range2_min = self.range2_min.replace(year = fecha_año)
        range2_max = self.range2_max.replace(year = fecha_año)
        range3_min = self.range3_min.replace(year = fecha_año)
        range3_max = self.range3_max.replace(year = fecha_año)
        range4_min = self.range4_min.replace(year = fecha_año)
        range4_max = self.range4_max.replace(year = fecha_año)
        
        if ((fecha_dt >= range1_min and fecha_dt <= range1_max) or 
            (fecha_dt >= range2_min and fecha_dt <= range2_max) or 
            (fecha_dt >= range3_min and fecha_dt <= range3_max) or
            (fecha_dt >= range4_min and fecha_dt <= range4_max)):
            return 1
        else:
            return 0
    
    def _get_period_day(self, date: str) -> str:
        """
        Get the period of the day for a given date. The period of the day is defined as follows:
        - Morning: 05:00 to 11:59
        - Afternoon: 12:00 to 18:59
        - Evening: 19:00 to 04:59
        Args:            
            date (str): date in the format 'YYYY-MM-DD HH:MM:SS'
        Returns:
            str: 'mañana' if the date is in the morning, 'tarde' if the date is in the afternoon, 'noche' if the date is in the evening.
        """
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        
        if date_time >= self.morning_min and date_time < self.afternoon_min:
            return 'mañana'
        elif date_time >= self.afternoon_min and date_time < self.evening_min:
            return 'tarde'
        else:
            return 'noche'
    
    def _get_min_diff(self, data: pd.Series) -> float:
        """
        Get the difference in minutes between the scheduled departure time and the actual departure time.
        Used to calculate the target variable.
        Args:
            data (pd.Series): a row of the dataframe with the columns 'Fecha-O' and 'Fecha-I' in the format 'YYYY-MM-DD HH:MM:SS'
        Returns:
            float: the difference in minutes between the scheduled departure time and the actual departure time.
        """
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff
    

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        
        # Validate that the expected features are in the data.
        if not set(self._EXPECTED_FEATURES).issubset(set(data.columns)):
            raise ValueError(f"Expected features {self._EXPECTED_FEATURES} not found in data columns {data.columns}")
        
        # Validate that the target column can be built
        if target_column and not set(['Fecha-O', 'Fecha-I']).issubset(set(data.columns)):
            raise ValueError(f"Expected features ['Fecha-O', 'Fecha-I'] not found in data columns {data.columns} to build target column {target_column}")
        
        # Create a copy of the data to avoid modifying the original dataframe.
        data = data.copy()

        # One-hot encode categorical features
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA', dtype=int),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO', dtype=int), 
            pd.get_dummies(data['MES'], prefix = 'MES', dtype=int)], 
            axis = 1
        )

        # If there's a missing category in the data, add it with all values set to 0.
        for feature in self._TOP_10_FEATURES:
            if feature not in features.columns:
                features[feature] = 0

        # Select only the top 10 features.
        features = features[self._TOP_10_FEATURES]

        if target_column:
            target = data.apply(self._get_min_diff, axis = 1)
            target = target.apply(lambda x: 1 if x > 15 else 0).to_frame(name = target_column)
            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        # Compute scale_pos_weight to handle class imbalance
        n_y0 = (target[target.columns[0]] == 0).sum()
        n_y1 = (target[target.columns[0]] == 1).sum()
        scale = n_y0/n_y1

        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
        self._model.fit(features, target)

        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.
        Requires features to be preprocessed with the preprocess method.
        Requires features to have self.top_10_features columns. Extra columns are allowed but will be ignored. Missing columns will raise an error.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        # Validate that the model has been fitted.
        if self._model is None:
            raise ValueError("Model has not been fitted yet. Please call the fit method before calling predict.")
        
        # Validate that the features have the expected columns.
        if not set(self._TOP_10_FEATURES).issubset(set(features.columns)):
            raise ValueError(f"Expected features {self._TOP_10_FEATURES} not found in features columns {features.columns}. Please preprocess the data using the preprocess method before calling predict.")
        
        predictions = self._model.predict(features[self._TOP_10_FEATURES])
        return predictions.tolist()