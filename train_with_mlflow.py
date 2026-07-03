import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

VOCAB_SIZE = 10000
MAX_LEN = 200

print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)

mlflow.set_experiment("sentiment-classification")


def run_v1():
    with mlflow.start_run(run_name="model_v1_baseline"):
        mlflow.log_param("model_version", "v1")
        mlflow.log_param("embedding_dim", 32)
        mlflow.log_param("lstm_units", 32)
        mlflow.log_param("epochs", 3)
        mlflow.log_param("batch_size", 64)

        model = Sequential([
            Embedding(VOCAB_SIZE, 32),
            LSTM(32),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2, verbose=1)

        test_loss, test_acc = model.evaluate(x_test, y_test)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)

        mlflow.keras.log_model(model, "model")
        model.save("model_v1.keras")
        print(f"v1 done: accuracy={test_acc:.4f}")


def run_v2():
    with mlflow.start_run(run_name="model_v2_improved"):
        mlflow.log_param("model_version", "v2")
        mlflow.log_param("embedding_dim", 64)
        mlflow.log_param("lstm_units", 24)
        mlflow.log_param("bidirectional", True)
        mlflow.log_param("l2_regularization", 0.001)
        mlflow.log_param("dropout", 0.5)
        mlflow.log_param("epochs", 10)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("early_stopping_patience", 1)

        model = Sequential([
            Embedding(VOCAB_SIZE, 64),
            Bidirectional(LSTM(24, kernel_regularizer=l2(0.001))),
            Dropout(0.5),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        early_stop = EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2,
                  callbacks=[early_stop], verbose=1)

        test_loss, test_acc = model.evaluate(x_test, y_test)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)

        mlflow.keras.log_model(model, "model")
        model.save("model_v2.keras")
        print(f"v2 done: accuracy={test_acc:.4f}")


if __name__ == "__main__":
    run_v1()
    run_v2()
    print("Both runs logged to MLflow.")
