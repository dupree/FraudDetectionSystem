import pandas as pd
from fraud_detector import FraudDetector

def main():
    # Load dataset
    df = pd.read_parquet("./data.parquet", engine="pyarrow") 

    # Initialize and train the detector
    detector = FraudDetector(feature_cache_dir="../features/")
    results = detector.train(df, validate=True, use_cache=True, num_rounds=300)

    # Simulate a real-world single transaction
    single_transaction = {
        "timestamp": "2025-02-17 12:30:00",  # Example timestamp
        "session_id": "abcd1234",
        "device": "mobile",
        "price": 150.00  # Example transaction price
    }

    prediction = detector.predict(single_transaction)

    # Print the prediction result
    print("\n=== Real-Time Fraud Detection ===")
    print(f"Is Fraudulent: {prediction['is_fraud']}")
    print(f"Fraud Probability: {prediction['fraud_probability']:.4f}")
    print(f"Confidence Score: {prediction['confidence']:.2f}")


if __name__ == "__main__":
    main()

