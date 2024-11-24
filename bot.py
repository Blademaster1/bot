import os
import logging
import ccxt
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import asyncio
import requests
import ta

# Claves API y webhook definidas directamente en el código
BINANCE_API_KEY = "tu_clave_api"
BINANCE_SECRET_KEY = "tu_clave_secreta"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/ID/TOKEN"

# Configuración inicial
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Configuración del logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("crypto_bot.log")]
)
logger = logging.getLogger()


class UnifiedCryptoBot:
    def __init__(self, api_key, secret_key, discord_webhook_url, timeframe="1h", retrain_interval=5, top_n=50):
        self.api_key = api_key
        self.secret_key = secret_key
        self.discord_webhook_url = discord_webhook_url
        self.timeframe = timeframe
        self.retrain_interval = retrain_interval  # Intervalo en minutos
        self.top_n = top_n
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.positions = {}
        self.model_path = "models/unified_model_{symbol}.keras"
        os.makedirs('models', exist_ok=True)
        self.exchange = self._initialize_exchange()

    def _initialize_exchange(self):
        """Configura el exchange Binance."""
        try:
            return ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'enableRateLimit': True
            })
        except Exception as e:
            logger.error(f"Error al inicializar el exchange: {e}")
            raise

    def send_notification(self, message):
        """Envía notificaciones al webhook de Discord."""
        try:
            requests.post(self.discord_webhook_url, json={"content": message})
        except Exception as e:
            logger.error(f"Error al enviar notificación a Discord: {e}")

    def fetch_historical_data(self, symbol, limit=1000):
        """Obtiene datos históricos desde Binance."""
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["datetime", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
            df.set_index("datetime", inplace=True)
            return df.dropna()
        except Exception as e:
            logger.error(f"Error al obtener datos para {symbol}: {e}")
            return pd.DataFrame()

    def add_advanced_indicators(self, df):
        """Añade indicadores técnicos."""
        try:
            df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
            df["ema_short"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
            df["ema_long"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
            return df.dropna()
        except Exception as e:
            logger.error(f"Error al añadir indicadores técnicos: {e}")
            return pd.DataFrame()

    def create_lstm_model(self, input_shape):
        """Crea un modelo LSTM."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def prepare_data_for_lstm(self, df):
        """Prepara datos para LSTM."""
        try:
            df_scaled = self.scaler.fit_transform(df[["close", "rsi", "ema_short", "ema_long"]])
            X, y = [], []
            for i in range(60, len(df_scaled)):
                X.append(df_scaled[i-60:i])
                y.append(df_scaled[i, 0])
            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(f"Error al preparar los datos: {e}")
            return None, None

    async def train_and_trade(self, symbol):
        """Entrena modelo y realiza operaciones."""
        try:
            df = self.fetch_historical_data(symbol)
            if df.empty or len(df) < 100:
                logger.warning(f"No hay suficientes datos para {symbol}")
                return

            df = self.add_advanced_indicators(df)
            X, y = self.prepare_data_for_lstm(df)

            if X is None or y is None or len(X) == 0 or len(y) == 0:
                logger.warning(f"No se pudieron preparar datos para {symbol}")
                return

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            input_shape = (X_train.shape[1], X_train.shape[2])

            model_path = self.model_path.format(symbol=symbol)
            if not os.path.exists(model_path):
                model = self.create_lstm_model(input_shape)
                early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
                model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss")
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stopping, model_checkpoint]
                )
            else:
                model = load_model(model_path)

            latest_data = X[-1:]
            prediction = model.predict(latest_data)[0][0]
            latest_price = df["close"].iloc[-1]

            if prediction > latest_price and symbol not in self.positions:
                self.positions[symbol] = latest_price
                message = f"Comprar {symbol} a {latest_price:.2f}"
                self.send_notification(message)
                logger.info(message)
            elif symbol in self.positions and latest_price > self.positions[symbol] * 1.02:
                buy_price = self.positions.pop(symbol)
                profit = latest_price - buy_price
                message = f"Vender {symbol} a {latest_price:.2f}, ganancia: {profit:.2f}"
                self.send_notification(message)
                logger.info(message)

        except Exception as e:
            logger.error(f"Error procesando {symbol}: {e}")

    async def run(self):
        """Ejecución principal."""
        while True:
            try:
                top_cryptos = [symbol for symbol in self.exchange.fetch_tickers().keys() if "/USDT" in symbol][:self.top_n]
                tasks = [self.train_and_trade(symbol) for symbol in top_cryptos]
                await asyncio.gather(*tasks)
                logger.info(f"Esperando {self.retrain_interval:.2f} minutos para el próximo ciclo...")
                await asyncio.sleep(self.retrain_interval * 60)  # Convertido a minutos
            except Exception as e:
                logger.error(f"Error crítico: {e}")
                break


if __name__ == "__main__":
    bot = UnifiedCryptoBot(BINANCE_API_KEY, BINANCE_SECRET_KEY, DISCORD_WEBHOOK_URL, retrain_interval=5)
    asyncio.run(bot.run())
