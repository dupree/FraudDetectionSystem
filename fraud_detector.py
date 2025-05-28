import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc, classification_report
from lightgbm.callback import early_stopping


class FraudDetector:
    def __init__(self, feature_cache_dir='features'):
        self.model = None
        self.threshold = 0.5
        self.session_history = {}
        self.feature_columns = None
        self.feature_cache_dir = feature_cache_dir

        # Create cache directory if it doesn't exist
        if not os.path.exists(feature_cache_dir):
            os.makedirs(feature_cache_dir)

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from raw data with custom features"""
        # Convert timestamp to datetime if string
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 4)).astype(int)

        # Session-based features
        features = [self._calculate_session_features(row) for _, row in df.iterrows()]

        # Convert features to DataFrame
        feature_df = pd.DataFrame(features)

        # Combine with original features
        final_features = pd.concat([
            df[['hour', 'day_of_week', 'is_weekend', 'is_night', 'price']],
            feature_df
        ], axis=1)

        # Store feature columns for prediction
        if self.feature_columns is None:
            self.feature_columns = final_features.columns.tolist()

        return final_features

    def _calculate_session_features(self, row: pd.Series) -> dict:
        """Calculate session-based features for a single row"""
        session_id = row['session_id']
        timestamp = row['timestamp']
        device = row['device']
        price = row['price']

        if session_id not in self.session_history:
            self.session_history[session_id] = {
                'last_timestamp': timestamp,
                'last_device': device,
                'order_count': 0,
                'device_switches': 0,
                'total_spent': 0,
                'min_price': price,
                'max_price': price,
                'prices': [price],
                'time_since_last_order': pd.Timedelta(seconds=0)
            }
            session_features = {
                'order_count': 1,
                'device_switches': 0,
                'total_spent': price,
                'avg_price': price,
                'price_variance': 0,
                'time_since_last_order_seconds': 0,
                'orders_per_hour': 0,
                'device_switch_rate': 0,
            }
        else:
            session = self.session_history[session_id]
            time_diff = timestamp - session['last_timestamp']
            device_switch = 1 if device != session['last_device'] else 0

            # Update session history
            session['order_count'] += 1
            session['device_switches'] += device_switch
            session['total_spent'] += price
            session['min_price'] = min(session['min_price'], price)
            session['max_price'] = max(session['max_price'], price)
            session['prices'].append(price)
            session['time_since_last_order'] = time_diff

            # Calculate features
            avg_price = session['total_spent'] / session['order_count']
            price_variance = np.var(session['prices']) if len(session['prices']) > 1 else 0
            orders_per_hour = session['order_count'] / max(time_diff.total_seconds() / 3600, 0.01)
            device_switch_rate = session['device_switches'] / session['order_count']

            session_features = {
                'order_count': session['order_count'],
                'device_switches': session['device_switches'],
                'total_spent': session['total_spent'],
                'avg_price': avg_price,
                'price_variance': price_variance,
                'time_since_last_order_seconds': time_diff.total_seconds(),
                'orders_per_hour': orders_per_hour,
                'device_switch_rate': device_switch_rate,
            }

            # Update session history for next order
            session['last_timestamp'] = timestamp
            session['last_device'] = device

        return session_features

    def _calculate_profit(self, y_true: np.ndarray, y_pred: np.ndarray, prices: np.ndarray, threshold: float) -> float:
        predictions = (y_pred >= threshold).astype(int)

        legitimate_accepted = (y_true == 0) & (predictions == 0)
        fraud_accepted = (y_true == 1) & (predictions == 0)

        profit = np.sum(prices[legitimate_accepted] * 0.01)
        loss = np.sum(prices[fraud_accepted])

        return profit - loss

    def _calculate_baseline_profit(self, y_true: np.ndarray, prices: np.ndarray) -> float:
        """Calculate the baseline profit assuming all transactions are accepted (no fraud detection)"""

        # Profit from all legitimate orders
        legitimate_profit = np.sum(prices[y_true == 0] * 0.01)  # 1% profit on legitimate orders

        # Loss from all fraudulent orders
        fraud_loss = np.sum(prices[y_true == 1])  # Full loss for fraudulent orders

        # Net baseline profit
        return legitimate_profit - fraud_loss


    def _optimize_threshold(self, y_true: np.ndarray, y_pred: np.ndarray, prices: np.ndarray) -> tuple[float, float]:
        """Optimize threshold by maximizing profit using few evaluation points."""
        thresholds = np.linspace(0.1, 0.9, 10)
        profits = [self._calculate_profit(y_true, y_pred, prices, t) for t in thresholds]

        # Select threshold that gives the highest profit
        optimal_threshold = thresholds[np.argmax(profits)]

        return optimal_threshold, max(profits)


    def train(self, df: pd.DataFrame, validate: bool = True, use_cache: bool = True, num_rounds: int = 51) -> dict:
        """Training with cross-validation and detailed metrics"""
        # Ensure cache directory exists
        os.makedirs(self.feature_cache_dir, exist_ok=True)
        feature_cache_path = os.path.join(self.feature_cache_dir, "cached_features.pkl")

        # Load cached features if available
        if use_cache and os.path.exists(feature_cache_path):
            with open(feature_cache_path, "rb") as f:
                print("Loading cached features...")
                X, self.feature_columns = pickle.load(f)
        else:
            print("Generating new features...")
            X = self._create_features(df)
            self.feature_columns = X.columns.tolist()

            # Save new features to cache
            with open(feature_cache_path, "wb") as f:
                pickle.dump((X, self.feature_columns), f)
                print("Features saved to cache.")

        y = df['is_fraud']
        prices = df['price']

        # Calculate class weights
        pos_weight = len(y[y==0]) / len(y[y==1])

        # Split data
        X_train, X_val, y_train, y_val, prices_train, prices_val = train_test_split(
            X, y, prices, test_size=0.2, random_state=42, stratify=y
        )

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        # Updated parameters
        params = {
            "objective": "binary",
            "metric": ['auc', 'average_precision'],
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "scale_pos_weight": pos_weight,  # Using only scale_pos_weight for imbalance
            "random_state": 42,
            'verbose': -1,
            'lambda_l1': 0.1,        # L1 regularization
            'lambda_l2': 0.1,         # L2 regularization
        }

        # Train with cross-validation if requested
        if validate:
            print("Performing cross-validation...")
            cv_results = lgb.cv(
                params,
                train_data,
                num_boost_round=num_rounds,
                nfold=5,
                callbacks=[early_stopping(stopping_rounds=50, verbose=True)],
                metrics=['auc', 'average_precision'],
                stratified=True,
                seed=42
            )

            best_rounds = len(cv_results.get('valid auc-mean', []))

            print(f"Best CV AUC: {cv_results['valid auc-mean'][-1]:.4f} ± {cv_results['valid auc-stdv'][-1]:.4f}")
            print(f"Best CV AP: {cv_results['valid average_precision-mean'][-1]:.4f} ± {cv_results['valid average_precision-stdv'][-1]:.4f}")
        else:
            best_rounds = num_rounds

        # Train final model
        print("\nTraining final model...")
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=best_rounds,
            callbacks=[early_stopping(stopping_rounds=50, verbose=True)]
        )

        # Generate and evaluate predictions
        val_pred = self.model.predict(X_val)

        # Calculate metrics
        auc_score = roc_auc_score(y_val, val_pred)
        ap_score = average_precision_score(y_val, val_pred)

        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_val, val_pred)

        # Compute AUC-PR
        auc_pr = auc(recall, precision)

        # Optimize threshold
        self.threshold, best_profit = self._optimize_threshold(y_val, val_pred, prices_val)

        baseline_profit = self._calculate_baseline_profit(y_val, prices_val)

        # Generate classification report
        val_pred_labels = (val_pred >= self.threshold).astype(int)
        clf_report = classification_report(y_val, val_pred_labels)

        # Print detailed metrics
        print("\nModel Evaluation:")
        print(f"AUC-ROC Score: {auc_score:.4f}")
        print(f"Average Precision Score: {ap_score:.4f}")
        print(f"AUC-PR Score: {auc_pr:.4f}")

        print(f"Optimal Threshold: {self.threshold:.4f}")
        print(f"Best Validation Profit: ${best_profit:.2f}")
        print(f"Baseline Profit (No ML Model): ${baseline_profit:.2f}")

        print("\nClassification Report:")
        print(clf_report)

        # Feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Important Features:")
        print(importance.head(10))

        return {
            'auc_score': auc_score,
            'ap_score': ap_score,
            'auc_pr' : auc_pr,
            'threshold': self.threshold,
            'best_profit': best_profit,
            'feature_importance': importance,
            'precision_recall_curve': (precision, recall, pr_thresholds)
        }

    def predict(self, order: dict) -> dict:
        """Make real-time prediction for a single order"""
        # Convert single order to DataFrame
        order_df = pd.DataFrame([order])

        # Create features
        X = self._create_features(order_df)

        # Ensure all features are present
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0

        # Reorder columns to match training data
        X = X[self.feature_columns]

        # Make prediction
        fraud_prob = self.model.predict(X)[0]

        # Decision based on threshold
        is_fraud = fraud_prob >= self.threshold

        return {
            'is_fraud': is_fraud,
            'fraud_probability': fraud_prob,
            'threshold_used': self.threshold,
            'confidence': abs(fraud_prob - 0.5) * 2  # Scale 0-1 where 1 is most confident
        }