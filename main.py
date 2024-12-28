import pandas as pd

from models.dcf_valuation import DCFValuation
from models.stock_predictor import StockPredictor

pd.set_option('future.no_silent_downcasting', True)


def main(ticker, model="random_forest"):
    """
    developed by Abdullah Alamri and Chun-Ju Tao.
    """
    # if True:
    try:
        predictor = StockPredictor(ticker)
        predictor.fetch_historical_data()
        predictor.calculate_technical_indicators()
        if "random_forest" in model:
            data, features = predictor.prepare_features()
            predictor.train_models_random_forest(features)
            predictions = predictor.predict_next_day_rf(features)

        if "LSTM" in model:
            data, features = predictor.prepare_features()
            predictor.train_models_LSTM(features)
            predictions = predictor.predict_next_day_LSTM(features)

        if "Transformer" in model:
            data, features, X_transformer, y_transformer = predictor.prepare_features_transformer(sequence_length=30)
            predictor.train_models_transformer(X_transformer, y_transformer)
            predictions_transformer = predictor.predict_next_day_transformer(sequence_length=30)
            print(f"\nTransformer Predictions for {ticker}:")
            print(f"Direction: {predictions_transformer['direction']}")
            # print(f"Confidence: {predictions_transformer['direction_confidence']:.1f}%") # Confidence is not directly available
            print(f"Predicted Price: ${predictions_transformer['predicted_price']:.2f}")
            print(
                f"Price Range: ${predictions_transformer['price_range'][0]:.2f} - ${predictions_transformer['price_range'][1]:.2f}")
        else:  # RF or LSTM
            print(f"\nTechnical Analysis Predictions for {ticker}:")
            print(f"Direction: {predictions['direction']} (Confidence: {predictions['direction_confidence']:.1f}%)")
            print(f"Predicted Price: ${predictions['predicted_price']:.2f}")
            print(f"Price Range: ${predictions['price_range'][0]:.2f} - ${predictions['price_range'][1]:.2f}")

        print("\nCalculating DCF Valuation...")
        dcf = DCFValuation(ticker)
        valuation = dcf.calculate_intrinsic_value()

        print(f"\nDCF Valuation Results:")
        print(f"Intrinsic Value: ${valuation['intrinsic_value']:.2f}")
        print(f"Current Price: ${valuation['current_price']:.2f}")
        print(f"Potential Upside: {valuation['upside']:.1f}%")
        print(f"WACC: {valuation['wacc']:.1f}%")
        print(f"Growth Rate: {valuation['growth_rate']:.1f}%")
        print(f"Terminal Growth: {valuation['terminal_growth']:.1f}%")

    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")


if __name__ == "__main__":
    main("MSFT", model="random_forest")
