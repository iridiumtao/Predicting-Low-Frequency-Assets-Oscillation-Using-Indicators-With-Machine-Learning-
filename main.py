import tensorflow as tf

def is_gpu_available():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs are available.")
        print(gpus)
        return True
    else:
        print("GPUs are not available.")
        return False


from config.config import Config
from data.data_handler import DataHandler
from models.model import Model
from training.trainer import Trainer
from predictions.predictor import Predictor
from predictions.visualization import display_results
from reports.report_generator import ReportGenerator


if __name__ == '__main__':
    is_gpu_available()

    # Load Config
    config = Config(
          start_date = "2023-05-21",
          end_date = "2023-12-13",
          prediction_date = "2023-12-22",
          tickers = [
              "AEFES", "AGHOL", "AHGAZ", "AKBNK", "AKCNS", "AKFGY", "AKFYE", "AKSA", "AKSEN", "ALARK",
              "ALBRK", "ALFAS", "ARCLK", "ASELS", "BERA", "BIENY", "BIMAS", "BRSAN", "BRYAT",
              "BUCIM", "CANTE", "CCOLA", "CIMSA", "CWENE", "DOAS", "DOHOL", "ECILC", "ECZYT", "EGEEN",
              "EKGYO", "ENJSA", "ENKAI", "EREGL", "EUPWR", "FROTO", "GARAN", "GENIL", "GESAN",
              "GLYHO", "GUBRF", "GWIND", "HALKB", "HEKTS", "IPEKE", "ISCTR", "ISDMR", "ISGYO",
              "ISMEN", "IZMDC", "KARSN", "KAYSE", "KCHOL", "KMPUR", "KONTR", "KONYA", "KORDS",
              "KOZAA", "KOZAL", "KRDMD", "KZBGY", "MAVI", "MGROS", "MIATK", "ODAS", "OTKAR", "OYAKC",
              "PENTA", "PETKM", "PGSUS", "PSGYO", "QUAGR", "SAHOL", "SASA", "SISE", "SKBNK", "SMRTG",
              "SNGYO", "SOKM", "TAVHL", "TCELL", "THYAO", "TKFEN", "TOASO", "TSKB", "TTKOM", "TTRAK",
              "TUKAS", "TUPRS", "ULKER", "VAKBN", "VESBE", "VESTL", "YKBNK", "YYLGD", "ZOREN"
              ],
          tickers_for_prediction = ["AKFYE.IS", "EUPWR.IS", "YYLGD.IS", "BIMAS.IS", "GENIL.IS", "ODAS.IS", "ISMEN.IS", "ZOREN.IS", "CANTE.IS", "AHGAZ.IS"]
    )
    tickers = [symbol + ".IS" for symbol in config.tickers]
    # Initialize DataHandler
    data_handler = DataHandler(start_date=config.start_date, end_date=config.end_date, tickers=tickers)

    # Initialize Model
    model = Model(model_type='lstm', output_type='regression')

    # Initialize Trainer
    trainer = Trainer(data_handler=data_handler, model=model, model_type = 'lstm')
    trainer.train_and_evaluate_all(add_custom_indicator=True, target_type='regression', correlated_asset='SPY')

    #Initialize Predictor
    data_handler.tickers = config.tickers_for_prediction
    predictor = Predictor(data_handler=data_handler, model=model, model_type = 'lstm')
    predictor.predict_for_tickers(prediction_date=config.prediction_date, add_custom_indicator=True, correlated_asset='SPY')

    # Display Results
    display_results(trainer.results_df, trainer.top_10_results, predictor.prediction_results)

    # Initialize Report Generator
    report_generator = ReportGenerator(trainer.top_10_results, trainer.results_df, predictor.prediction_results)

    # Save the results in csv files
    report_generator.save_results_to_csv()

