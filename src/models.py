"""Modele prognozujące zwrot na następny dzień: wspólny interfejs fit/predict.

Co robi: definiuje klasę bazową `BaseModel` i trzy modele (naiwny, ARIMA, LSTM) oraz wybór rzędu ARIMA.
Dlaczego istnieje: silnik walk-forward (`walk_forward.py`) ma uczyć i oceniać każdy model tak samo, nie wiedząc,
z którym pracuje; wspólny interfejs (jak klasa bazowa i dziedziczenie w C#) to umożliwia i pozwala dołożyć
nowy model bez zmian w pętli.

Każdy model dostaje w `fit` tabelę cech z dni uczących, a w `predict` tabelę cech
z dni testowych (ten sam format co z `features.build_feature_table`) i zwraca
tablicę prognozowanych zwrotów, po jednej dla każdego dnia testowego.
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config


class BaseModel:
    """Klasa bazowa: każdy model ma nazwę, `fit`, `predict` i liczbę parametrów.

    `fit` i `predict` zgłaszają `NotImplementedError`, żeby zapomnienie o implementacji w podklasie
    było natychmiast widoczne, a nie dawało po cichu złych prognoz.
    """

    name = "base"

    def __init__(self):
        self.fit_seconds = 0.0

    def fit(self, train_data: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def n_parameters(self) -> int:
        return 0

    def timed_fit(self, train_data: pd.DataFrame) -> None:
        """Wywołuje `fit` i zapamiętuje czas uczenia w sekundach."""
        start = time.perf_counter()
        self.fit(train_data)
        self.fit_seconds = time.perf_counter() - start


class NaiveModel(BaseModel):
    """Model naiwny: prognoza zwrotu zawsze 0 (czyli kierunek: zawsze wzrost).

    Odpowiada prognozie "jutro cena taka jak dziś". Jest punktem odniesienia: model, który nie jest
    od niego lepszy, nie wnosi informacji. `fit` nic nie robi, bo nie ma czego się uczyć.
    """

    name = "naive"

    def fit(self, train_data: pd.DataFrame) -> None:
        pass

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(test_data))


class ArimaModel(BaseModel):
    """ARIMA(p, 0, q) na dziennych zwrotach logarytmicznych.

    Rząd (p, q) jest wybierany raz (kryterium AIC, lata 2015–2019) i podawany w konstruktorze.
    Parametry są estymowane w `fit` na danych uczących okna. W `predict` dla każdego dnia
    testowego: dołączamy zaobserwowany zwrot tego dnia (`append(refit=False)`, czyli bez
    ponownej estymacji parametrów) i prognozujemy zwrot następnego dnia.
    """

    name = "arima"

    def __init__(self, order: tuple[int, int, int]):
        super().__init__()
        self.order = order
        self.results = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """Estymuje parametry ARIMA na zwrotach uczących (metodą największej wiarygodności)."""
        from statsmodels.tsa.arima.model import ARIMA  # import wewnątrz: statsmodels nie jest potrzebny aplikacji

        returns = train_data["log_return"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results = ARIMA(returns, order=self.order).fit()

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Prognozy jednodniowe dla każdego dnia testowego z aktualizacją stanu, bez ponownej estymacji."""
        returns = test_data["log_return"].to_numpy()
        predictions = np.empty(len(returns))
        results = self.results  # kopia referencji: `append` zwraca nowy obiekt, oryginał zostaje nietknięty
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, observed_return in enumerate(returns):
                # Po zamknięciu dnia t znamy zwrot dnia t; prognozujemy zwrot dnia t+1.
                results = results.append(np.array([observed_return]), refit=False)
                predictions[i] = float(results.forecast(1)[0])
        return predictions

    def n_parameters(self) -> int:
        """Liczba parametrów (stała, współczynniki AR/MA, wariancja); dla rzędu (0,0,0) są to 2 parametry."""
        return int(len(self.results.params)) if self.results is not None else 0

    def save(self, path: Path) -> None:
        """Zapisuje rząd i wyestymowane parametry do pliku JSON (czytelnego dla człowieka).

        Sam JSON nie wystarczy do prognozowania: ARIMA potrzebuje też szeregu uczącego, żeby
        odtworzyć stan filtru. Dlatego `load` wymaga podania tych samych danych uczących.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = {
            "order": list(self.order),
            "param_names": list(self.results.param_names),
            "params": [float(value) for value in self.results.params],
            "n_train_days": int(self.results.nobs),
            "aic": float(self.results.aic),
            "log_likelihood": float(self.results.llf),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path, train_data: pd.DataFrame) -> "ArimaModel":
        """Wczytuje parametry z JSON i odtwarza model na tych samych danych uczących (bez estymacji).

        `filter(params)` buduje wyniki ARIMA z gotowymi parametrami, więc stan po danych uczących
        jest identyczny jak po `fit`, a prognozy są te same (sprawdza to test).
        """
        from statsmodels.tsa.arima.model import ARIMA

        with open(path, encoding="utf-8") as handle:
            content = json.load(handle)
        model = cls(order=tuple(content["order"]))
        returns = train_data["log_return"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.results = ARIMA(returns, order=model.order).filter(np.array(content["params"]))
        return model


def select_arima_order(returns: pd.Series, candidates=config.ARIMA_ORDER_SEARCH) -> tuple[dict, pd.DataFrame]:
    """Wybiera (p, q) o najniższym AIC spośród kandydatów (d = 0). Zwraca wybór i tabelę wszystkich wyników.

    AIC nagradza dopasowanie i karze liczbę parametrów. Wywoływane raz, na latach 2015–2019, żeby wybór rzędu
    nie korzystał z okresu testowego. Wynik (0, 0, 0) oznacza, że zwroty zachowują się jak biały szum wokół stałej.
    """
    from statsmodels.tsa.arima.model import ARIMA

    rows = []
    values = returns.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in candidates:
            for q in candidates:
                fitted = ARIMA(values, order=(p, 0, q)).fit()
                rows.append({"p": p, "d": 0, "q": q, "aic": float(fitted.aic), "bic": float(fitted.bic)})
    table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    best = table.iloc[0]
    chosen = {"p": int(best["p"]), "d": 0, "q": int(best["q"]), "aic": float(best["aic"])}
    return chosen, table


def make_sequences(features: np.ndarray, window_size: int) -> np.ndarray:
    """Tworzy tablicę sekwencji (liczba sekwencji × window_size × liczba cech).

    Sekwencja numer i kończy się na wierszu i + window_size - 1 i zawiera `window_size`
    kolejnych dni wstecz (włącznie z tym dniem). Nie sięga ani jednego dnia w przód.
    """
    n_sequences = len(features) - window_size + 1
    if n_sequences <= 0:
        return np.empty((0, window_size, features.shape[1]))
    return np.stack([features[i : i + window_size] for i in range(n_sequences)])


class LstmModel(BaseModel):
    """Sieć LSTM (Keras) prognozująca zwrot na następny dzień z okna ostatnich dni.

    Architektura: wejście (okno × cechy) -> 1 lub 2 warstwy LSTM -> Dropout -> Dense(1).
    Uczenie: Adam, strata MAE, batch 32, maks. 50 epok, EarlyStopping(patience=5, restore_best_weights).
    Walidacja: ostatnie 20% dni uczących (chronologicznie). Skalery (StandardScaler dla cech i celu)
    dopasowane tylko na pierwszych 80% dni uczących. W `predict` okno na początku kwartału sięga
    do ostatnich dni uczących (to przeszłość, więc bez wycieku).
    """

    name = "lstm"

    def __init__(
        self,
        feature_names: list[str],
        window_size: int,
        units: int,
        n_layers: int,
        seed: int = 0,
        max_epochs: int = config.LSTM_MAX_EPOCHS,
        batch_size: int = config.LSTM_BATCH_SIZE,
        patience: int = config.LSTM_PATIENCE,
        dropout: float = config.LSTM_DROPOUT,
        validation_fraction: float = config.LSTM_VALIDATION_FRACTION,
    ):
        super().__init__()
        self.feature_names = feature_names
        self.window_size = window_size
        self.units = units
        self.n_layers = n_layers
        self.seed = seed
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout
        self.validation_fraction = validation_fraction
        self.feature_scaler = None
        self.target_scaler = None
        self.network = None
        self.train_tail = None  # ostatnie dni uczące potrzebne do pierwszych okien testowych
        self.epochs_trained = 0

    def _build_network(self, n_features: int):
        """Buduje sieć: wejście (okno × cechy) -> 1–2 warstwy LSTM -> Dropout -> Dense(1).

        Mała sieć celowo: przy ~1200–2700 dniach uczących większa tylko szybciej się przeucza.
        Strata MAE zamiast MSE, bo zwroty mają grube ogony i MSE kazałoby sieci "gonić" pojedyncze dni ±10%.
        """
        import keras  # import wewnątrz: TensorFlow ładuje się kilka sekund i nie jest potrzebny aplikacji

        layers = [keras.layers.Input(shape=(self.window_size, n_features))]
        for layer_index in range(self.n_layers):
            is_last_lstm = layer_index == self.n_layers - 1
            # Każda warstwa LSTM poza ostatnią musi oddawać całą sekwencję, żeby następna miała co czytać.
            layers.append(keras.layers.LSTM(self.units, return_sequences=not is_last_lstm))
        layers.append(keras.layers.Dropout(self.dropout))  # losowe wyłączanie neuronów w uczeniu (regularyzacja)
        layers.append(keras.layers.Dense(1))  # jeden neuron liniowy: prognozowany (przeskalowany) zwrot
        network = keras.Sequential(layers)
        network.compile(optimizer="adam", loss="mae")  # Adam z domyślnym tempem uczenia
        return network

    def fit(self, train_data: pd.DataFrame) -> None:
        """Uczy sieć na dniach uczących okna z wczesnym zatrzymaniem na ostatnich 20% dni.

        Kolejność ma znaczenie dla braku wycieku: najpierw podział 80/20 po wierszach (chronologicznie),
        potem skalery dopasowane TYLKO na pierwszych 80%, dopiero potem przekształcenie całości i cięcie na sekwencje.
        """
        import keras
        from sklearn.preprocessing import StandardScaler

        set_global_seed(self.seed)  # ten sam seed -> identyczne wagi początkowe, kolejność i dropout -> identyczny wynik

        # Podział chronologiczny: pierwsze 80% dni do uczenia wag, ostatnie 20% tylko do wczesnego zatrzymania.
        n_fit_rows = int(round(len(train_data) * (1 - self.validation_fraction)))
        fit_rows = train_data.iloc[:n_fit_rows]
        self.feature_scaler = StandardScaler().fit(fit_rows[self.feature_names].to_numpy())
        self.target_scaler = StandardScaler().fit(fit_rows[[config.TARGET]].to_numpy())

        # Skalery znają tylko średnią i odchylenie z części uczącej; walidację i test tylko przekształcają.
        features_scaled = self.feature_scaler.transform(train_data[self.feature_names].to_numpy())
        target_scaled = self.target_scaler.transform(train_data[[config.TARGET]].to_numpy()).ravel()

        sequences = make_sequences(features_scaled, self.window_size)
        # Sekwencja i kończy się na wierszu i + window_size - 1, więc jej celem jest target tego właśnie wiersza.
        targets = target_scaled[self.window_size - 1 :]
        # Sekwencja numer i kończy się na wierszu i + window_size - 1; do walidacji idą te,
        # których ostatni dzień należy do ostatnich 20% dni uczących.
        n_fit_sequences = max(n_fit_rows - self.window_size + 1, 0)
        x_fit, y_fit = sequences[:n_fit_sequences], targets[:n_fit_sequences]
        x_val, y_val = sequences[n_fit_sequences:], targets[n_fit_sequences:]

        self.network = self._build_network(len(self.feature_names))
        callbacks = []
        validation = None
        if len(x_val) > 0:
            validation = (x_val, y_val)
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self.patience, restore_best_weights=True
                )
            )
        history = self.network.fit(
            x_fit,
            y_fit,
            validation_data=validation,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            shuffle=True,  # tasujemy SEKWENCJE między paczkami; każda sekwencja i tak zawiera tylko własną przeszłość
            callbacks=callbacks,
            verbose=0,
        )
        self.epochs_trained = len(history.history["loss"])  # ile epok faktycznie trwało uczenie (do raportu kosztu)
        # Ostatnie window_size - 1 dni uczących: pierwsze dni kwartału potrzebują ich, żeby mieć pełne okno (to przeszłość).
        self.train_tail = train_data.iloc[-(self.window_size - 1) :] if self.window_size > 1 else train_data.iloc[0:0]

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Prognoza dla każdego dnia testowego z okna kończącego się na tym dniu (bez sięgania w przód)."""
        combined = pd.concat([self.train_tail, test_data])  # doklejamy przeszłość, nigdy przyszłość
        features_scaled = self.feature_scaler.transform(combined[self.feature_names].to_numpy())
        sequences = make_sequences(features_scaled, self.window_size)
        # Pierwsza sekwencja kończy się na pierwszym dniu testowym, ostatnia na ostatnim.
        assert len(sequences) == len(test_data)
        scaled_predictions = self.network.predict(sequences, batch_size=self.batch_size, verbose=0)
        return self.target_scaler.inverse_transform(scaled_predictions).ravel()  # z powrotem do jednostek zwrotu

    def n_parameters(self) -> int:
        """Liczba wag sieci (miara złożoności modelu do tabeli kosztów)."""
        return int(self.network.count_params()) if self.network is not None else 0

    def save(self, path_stem: Path) -> None:
        """Zapisuje wytrenowany model w dwóch plikach obok siebie.

        `<stem>.keras` to sieć (architektura + wagi), a `<stem>_scalers.joblib` to wszystko,
        co jest potrzebne, żeby z tej sieci odtworzyć prognozy: oba skalery, lista cech, ustawienia
        i ostatnie dni uczące (`train_tail`), bez których pierwsze dni kwartału nie miałyby pełnego okna.
        """
        path_stem = Path(path_stem)
        path_stem.parent.mkdir(parents=True, exist_ok=True)
        self.network.save(path_stem.parent / f"{path_stem.name}.keras")
        bundle = {
            "feature_names": list(self.feature_names),
            "window_size": self.window_size,
            "units": self.units,
            "n_layers": self.n_layers,
            "seed": self.seed,
            "epochs_trained": self.epochs_trained,
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "train_tail": self.train_tail,
        }
        joblib.dump(bundle, path_stem.parent / f"{path_stem.name}_scalers.joblib")

    @classmethod
    def load(cls, path_stem: Path) -> "LstmModel":
        """Wczytuje model zapisany przez `save`; gotowy do `predict` bez ponownego uczenia."""
        import keras

        path_stem = Path(path_stem)
        bundle = joblib.load(path_stem.parent / f"{path_stem.name}_scalers.joblib")
        model = cls(
            feature_names=bundle["feature_names"],
            window_size=bundle["window_size"],
            units=bundle["units"],
            n_layers=bundle["n_layers"],
            seed=bundle["seed"],
        )
        model.epochs_trained = bundle["epochs_trained"]
        model.feature_scaler = bundle["feature_scaler"]
        model.target_scaler = bundle["target_scaler"]
        model.train_tail = bundle["train_tail"]
        model.network = keras.saving.load_model(path_stem.parent / f"{path_stem.name}.keras")
        return model


def set_global_seed(seed: int) -> None:
    """Ustawia ziarna losowości (random, numpy, tensorflow) i wymusza deterministyczne operacje TF.

    Bez tego dwa uczenia z tymi samymi danymi dawałyby różne wagi (losowa inicjalizacja, tasowanie, dropout),
    a wyniki w pracy nie dałyby się odtworzyć. Sprawdzono: ten sam seed daje identyczne prognozy (DZIENNIK.md).
    """
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(seed)  # ustawia random, numpy i backend (tensorflow)
    tf.config.experimental.enable_op_determinism()
