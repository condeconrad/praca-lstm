"""Aplikacja Streamlit: przegląd danych, prognoz, porównania modeli i symulacji inwestowania.

Dlaczego osobna aplikacja: wyniki eksperymentu powstają raz (run_experiment.py, kilkadziesiąt minut),
a oglądać chcemy je wiele razy i natychmiast, także na obronie. Dlatego aplikacja NICZEGO nie trenuje:
tylko czyta pliki z `data/raw/` i `results/` i je pokazuje. Jedyne obliczenia "na żywo" to ponowne
przeliczenie strategii inwestycyjnych suwakami (prowizja, próg), bo to kilka linijek numpy.

Uruchomienie: `.venv/bin/streamlit run app/app.py`
Wygląd: motyw w `.streamlit/config.toml`, tylko wbudowane komponenty Streamlit, kilka linijek CSS niżej.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.strategy import evaluate_strategies  # noqa: E402  (tylko numpy i pandas, bez TensorFlow)

DATA_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_DIR = PROJECT_DIR / "results"

# Kolory: jeden akcent (granat) dla głównej serii, szarości dla pozostałych.
ACCENT = "#1F3A5F"
GRAY = "#8C8C8C"
LIGHT_GRAY = "#C9C9C9"
DARK_GRAY = "#4B4B4B"
MODEL_LABELS = {"naive": "Naiwny", "arima": "ARIMA", "lstm_A": "LSTM A", "lstm_B": "LSTM B"}
MODEL_COLORS = {"naive": ACCENT, "arima": DARK_GRAY, "lstm_A": GRAY, "lstm_B": LIGHT_GRAY}
STRATEGY_LABELS = {"buy_and_hold": "Kup i trzymaj", "basic": "Podstawowa", "threshold": "Z progiem"}

# Kilka linijek CSS w jednym miejscu: szerokość kolumny dopasowana do ekranu laptopa, odstępy, wersaliki w podpisach.
STYLE = """
<style>
.block-container {max-width: 1180px; padding-left: 3rem; padding-right: 3rem; padding-bottom: 4rem;}
[data-testid="stCaptionContainer"] p {letter-spacing: 0.08em; text-transform: uppercase; font-size: 0.72rem; color: #6B6B6B;}
[data-testid="stMetricLabel"] p {font-size: 0.8rem; color: #6B6B6B;}
h1 {letter-spacing: -0.02em; margin-bottom: 0.2rem;}
</style>
"""

# Krótkie objaśnienia pojęć. Te same teksty trafiają do dymków `help=` przy liczbach i do słowniczków w zakładkach.
GLOSSARY = {
    "Zwrot logarytmiczny": "ln(cena dziś / cena wczoraj). Dla małych zmian prawie równy zwykłemu zwrotowi procentowemu, "
                           "ale wygodniejszy matematycznie (zwroty z kilku dni się dodają). 0,01 oznacza ok. +1%.",
    "Cel prognozy": "Zwrot logarytmiczny z dnia t na dzień t+1, czyli 'o ile zmieni się cena jutro'. Prognoza powstaje "
                    "po zamknięciu sesji dnia t wyłącznie z danych znanych do tego dnia.",
    "Kierunek": "Wzrost, gdy zwrot jest >= 0; w przeciwnym razie spadek. Ta sama reguła dla prognoz i dla rzeczywistości.",
    "Trafność kierunku": "Odsetek dni, w których model poprawnie przewidział, czy cena wzrośnie, czy spadnie. 50% to poziom "
                         "rzutu monetą; model naiwny osiąga tyle, ile wynosi odsetek dni wzrostowych.",
    "MAE": "Średni błąd bezwzględny: średnia z |prognoza - rzeczywistość|. Im mniejszy, tym lepiej. Wyrażony w jednostkach "
           "zwrotu logarytmicznego (0,024 to ok. 2,4 punktu procentowego dziennie).",
    "RMSE": "Pierwiastek ze średniego błędu kwadratowego. Podobny do MAE, ale duże pomyłki ważą więcej, więc jest "
            "wrażliwszy na dni z gwałtownymi ruchami.",
    "Odsetek prognoz wzrostu": "Jaka część prognoz modelu jest >= 0. 100% oznacza, że model zawsze przewiduje wzrost "
                               "(tak działa model naiwny).",
    "Odchylenie std prognoz": "Rozrzut prognoz wokół ich średniej. Wartość bliska zera oznacza, że model prognozuje "
                              "prawie stałą liczbę, czyli w praktyce kopiuje model naiwny ('zapadł się').",
    "Model naiwny": "Prognoza zwrotu zawsze równa 0 ('jutro bez zmian'), czyli kierunek zawsze 'wzrost'. Punkt odniesienia: "
                    "model, który nie jest od niego lepszy, nie wnosi informacji.",
    "ARIMA": "Klasyczny model statystyczny szeregów czasowych, który przewiduje następną wartość z kilku poprzednich "
             "wartości i poprzednich błędów. Rząd (p, d, q) wybrano kryterium AIC na latach 2015-2019.",
    "LSTM": "Sieć neuronowa rekurencyjna z pamięcią (Long Short-Term Memory). Dostaje okno ostatnich dni (np. 20) i uczy się "
            "przewidywać jutrzejszy zwrot. Wariant A widzi tylko zwrot i zmianę wolumenu, wariant B także wskaźniki techniczne.",
    "Test dwumianowy": "Sprawdza, czy trafność kierunku jest istotnie większa niż 50%. Wartość p poniżej 0,05 oznacza, że "
                       "taki wynik rzadko zdarza się przypadkiem. Uwaga: w okresie wzrostowym nawet model naiwny go zdaje.",
    "Test McNemara": "Porównuje dwa modele na tych samych dniach: liczy dni, w których tylko jeden z nich trafił, i sprawdza "
                     "testem dwumianowym, czy badany model wygrywa częściej niż model naiwny. p < 0,05 oznaczałoby "
                     "istotną przewagę nad modelem naiwnym.",
    "Wartość p": "Prawdopodobieństwo uzyskania co najmniej tak dobrego wyniku, gdyby model naprawdę nie miał przewagi. "
                 "Przyjęty poziom istotności to 0,05.",
    "Dni nietypowe": "Dni wskazane przez algorytm Isolation Forest (drzewa losowo dzielące dane; obserwacje łatwe do "
                     "odizolowania są nietypowe). Cechy: |zwrot|, zmiana wolumenu, zmienność z 20 dni. Model dopasowany "
                     "tylko na danych uczących okna, próg 2% na tych danych.",
    "Walk-forward": "Schemat uczciwej oceny: na początku każdego kwartału model uczy się na wszystkich danych do dnia przed "
                    "kwartałem, potem prognozuje każdy dzień kwartału. 24 kwartały 2020-2025, te same dni dla wszystkich modeli.",
    "Ziarno": "Liczba startowa generatora losowego. Sieć LSTM uczona 5 razy z ziarnami 0-4 daje nieco inne prognozy; "
              "raportujemy każde ziarno i ich średnią, żeby pokazać, jak wynik zależy od losowości.",
    "Zwrot całkowity": "O ile procent zmieniła się wartość portfela od początku do końca okresu testowego (start = 1,0).",
    "CAGR": "Średnioroczna stopa zwrotu: (wartość końcowa / początkowa)^(252 / liczba dni) - 1. 252 to liczba sesji w roku.",
    "Maksymalne obsunięcie": "Największy spadek wartości portfela od szczytu do dołka w okresie. -66% oznacza, że w pewnym "
                             "momencie portfel był wart o dwie trzecie mniej niż na szczycie.",
    "Wskaźnik Sharpe'a": "Średni dzienny zwrot strategii podzielony przez jego odchylenie standardowe, w skali rocznej "
                         "(razy pierwiastek z 252), przy stopie wolnej od ryzyka 0. Mierzy zysk na jednostkę ryzyka.",
    "Transakcje": "Liczba wejść i wyjść z pozycji. Każda kosztuje prowizję (domyślnie 0,1% wartości portfela).",
    "Dni w akcjach": "Odsetek dni testowych, w których strategia trzymała akcje (a nie gotówkę).",
    "Kup i trzymaj": "Kupno akcji pierwszego dnia okresu testowego i sprzedaż ostatniego. Płaci prowizję za wejście i wyjście.",
    "Strategia podstawowa": "W akcjach, gdy prognoza na jutro jest >= 0; w przeciwnym razie gotówka. Dla modelu naiwnego "
                            "daje dokładnie 'kup i trzymaj'.",
    "Strategia z progiem": "W akcjach tylko wtedy, gdy prognoza przekracza próg (domyślnie 0,1%). Ma unikać wchodzenia "
                           "na rynek przy prognozach bliskich zera.",
    "Skośność": "Miara asymetrii rozkładu zwrotów. Ujemna: duże spadki zdarzają się częściej niż równie duże wzrosty.",
    "Kurtoza": "Miara 'grubości ogonów' rozkładu (nadwyżkowa, czyli ponad rozkład normalny). Wysoka oznacza, że ekstremalne "
               "dni zdarzają się znacznie częściej, niż zakłada rozkład normalny.",
}


def glossary(*terms: str) -> None:
    """Rozwijany słowniczek z wybranymi pojęciami (ten sam tekst co w dymkach `help`)."""
    with st.expander("Co oznaczają te pojęcia"):
        for term in terms:
            st.markdown(f"**{term}.** {GLOSSARY[term]}")


# --- Wczytywanie plików ------------------------------------------------------

@st.cache_data
def load_csv(name: str, parse_date: bool = True) -> pd.DataFrame:
    """Wczytuje plik CSV z results/ (z pamięcią podręczną Streamlit, żeby nie czytać dysku przy każdym kliknięciu)."""
    frame = pd.read_csv(RESULTS_DIR / name)
    if parse_date and "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


@st.cache_data
def load_prices() -> pd.DataFrame:
    """Wczytuje ceny NVDA i dolicza dzienny zwrot logarytmiczny do statystyk w zakładce Dane."""
    prices = pd.read_csv(DATA_DIR / "NVDA.csv", parse_dates=["Date"], index_col="Date")
    prices["log_return"] = np.log(prices["Close"] / prices["Close"].shift(1))
    return prices


@st.cache_data
def load_json(name: str) -> dict:
    """Wczytuje plik JSON z results/ (np. run_info.json)."""
    with open(RESULTS_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def results_available() -> bool:
    """Czy jest co pokazać: bez prognoz i run_info aplikacja pokazuje tylko zakładkę Dane z ostrzeżeniem."""
    return (RESULTS_DIR / "predictions.csv").exists() and (RESULTS_DIR / "run_info.json").exists()


# --- Pomocnicze: wykresy i formatowanie -----------------------------------------

def base_figure(y_title: str, height: int = 420) -> go.Figure:
    """Pusty wykres w stylu aplikacji: szablon plotly_white, bez siatki pionowej, legenda nad wykresem."""
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        font=dict(family="Inter, -apple-system, Helvetica Neue, sans-serif", color="#2B2B2B"),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(title=y_title, gridcolor="#EEEEEE", zerolinecolor="#DDDDDD")
    return figure


def percent(value: float) -> str:
    """Procent z jednym miejscem po przecinku; bardzo duże wartości (np. +3056%) bez miejsc, żeby mieściły się w kafelku."""
    if np.isnan(value):
        return "–"
    if abs(value) >= 10:
        return f"{value * 100:+.0f}%"
    return f"{value * 100:.1f}%"


def metric(column, label: str, value: str, term: str | None = None) -> None:
    """st.metric z dymkiem objaśniającym pojęcie ze słowniczka."""
    column.metric(label, value, help=GLOSSARY.get(term or label))


# --- Zakładka 1: Dane -----------------------------------------------------------

def tab_data(prices: pd.DataFrame, anomalies: pd.DataFrame | None, descriptive: pd.DataFrame | None) -> None:
    """Zakładka 1: cena NVDA z dniami nietypowymi i statystyki zwrotów w wybranym okresie."""
    st.caption("Cena zamknięcia NVDA z zaznaczonymi dniami nietypowymi")
    st.write("Wykres pokazuje skorygowaną cenę akcji NVIDIA w wybranym okresie; punkty to dni, "
             "które algorytm Isolation Forest uznał za nietypowe.")

    periods = {"Cały okres": ("2015-01-01", "2026-01-31"),
               "2015–2019 (uczenie i strojenie)": ("2015-01-01", "2019-12-31"),
               "2020–2025 (test)": ("2020-01-01", "2025-12-31")}
    for year in range(2015, 2026):
        periods[str(year)] = (f"{year}-01-01", f"{year}-12-31")
    choice = st.selectbox("Okres", list(periods.keys()), index=0,
                          help="Lata 2015–2019 służyły do uczenia i strojenia, lata 2020–2025 to okres testowy.")
    start, end = periods[choice]
    part = prices[(prices.index >= start) & (prices.index <= end)]

    figure = base_figure("Cena [USD]")
    figure.add_trace(go.Scatter(x=part.index, y=part["Close"], mode="lines", name="Cena zamknięcia",
                                line=dict(color=ACCENT, width=1.5)))
    flagged_frames = []
    if anomalies is not None:
        flagged_frames.append(anomalies[anomalies["is_anomaly"]][["date"]])
    if descriptive is not None:
        flagged_frames.append(descriptive[descriptive["is_anomaly"]][["date"]])
    if flagged_frames:
        flagged = pd.concat(flagged_frames)
        flagged = flagged[(flagged["date"] >= start) & (flagged["date"] <= end)]
        flagged = flagged.join(part["Close"], on="date", how="inner")
        figure.add_trace(go.Scatter(x=flagged["date"], y=flagged["Close"], mode="markers", name="Dzień nietypowy",
                                    marker=dict(color=GRAY, size=6)))
    st.plotly_chart(figure, width="stretch")
    if descriptive is not None:
        st.write("Dni nietypowe w latach 2015–2019 pochodzą z analizy opisowej (model dopasowany na tych samych latach) "
                 "i nie są używane w ocenie modeli. Dni testowe 2020–2025 oznacza model dopasowany tylko na danych "
                 "uczących danego okna, więc odsetek nie musi wynosić 2%.")

    st.caption("Statystyki dziennych zwrotów logarytmicznych w wybranym okresie")
    returns = part["log_return"].dropna()
    columns = st.columns(4)
    metric(columns[0], "Liczba dziennych zwrotów", f"{len(returns)}", None)
    metric(columns[1], "Średni dzienny zwrot", percent(returns.mean()), "Zwrot logarytmiczny")
    metric(columns[2], "Odchylenie standardowe", percent(returns.std()), "Odchylenie std prognoz")
    metric(columns[3], "Odsetek dni wzrostowych", percent((returns >= 0).mean()), "Kierunek")
    columns = st.columns(4)
    metric(columns[0], "Największy spadek", percent(returns.min()), "Zwrot logarytmiczny")
    metric(columns[1], "Największy wzrost", percent(returns.max()), "Zwrot logarytmiczny")
    metric(columns[2], "Skośność", f"{returns.skew():.2f}")
    metric(columns[3], "Kurtoza (nadwyżkowa)", f"{returns.kurt():.2f}", "Kurtoza")
    glossary("Zwrot logarytmiczny", "Kierunek", "Skośność", "Kurtoza", "Dni nietypowe")


# --- Zakładka 2: Prognozy ----------------------------------------------------------

def tab_predictions(predictions: pd.DataFrame) -> None:
    """Zakładka 2: prognozy wybranego modelu na tle rzeczywistych zwrotów (miary liczone jak w src/metrics.py)."""
    st.caption("Prognozy wybranego modelu na tle rzeczywistych zwrotów")
    st.write("Każdy punkt to prognoza zwrotu na następny dzień, wykonana po zamknięciu sesji wyłącznie z danych "
             "dostępnych do tego dnia. Prognozy LSTM to średnia z 5 ziaren.")
    models = [m for m in MODEL_LABELS if m in predictions["model"].unique()]
    columns = st.columns(2)
    model = columns[0].selectbox("Model", models, format_func=MODEL_LABELS.get, help=GLOSSARY["LSTM"])
    years = sorted(predictions["date"].dt.year.unique())
    year_choice = columns[1].selectbox("Okres", ["Cały okres testowy"] + [str(y) for y in years])
    part = predictions[predictions["model"] == model].sort_values("date")
    if year_choice != "Cały okres testowy":
        part = part[part["date"].dt.year == int(year_choice)]

    hits = (part["y_true"] >= 0) == (part["y_pred"] >= 0)
    columns = st.columns(4)
    metric(columns[0], "Trafność kierunku", percent(hits.mean()))
    metric(columns[1], "MAE", f"{(part['y_pred'] - part['y_true']).abs().mean():.4f}")
    metric(columns[2], "Odsetek prognoz wzrostu", percent((part["y_pred"] >= 0).mean()))
    # ddof=0 jak w src/metrics.py, żeby liczba zgadzała się z metrics.csv.
    metric(columns[3], "Odchylenie std prognoz", f"{part['y_pred'].std(ddof=0):.4f}")

    figure = base_figure("Zwrot logarytmiczny")
    figure.add_trace(go.Scatter(x=part["date"], y=part["y_true"], mode="lines", name="Rzeczywisty zwrot",
                                line=dict(color=LIGHT_GRAY, width=1)))
    figure.add_trace(go.Scatter(x=part["date"], y=part["y_pred"], mode="lines", name=f"Prognoza: {MODEL_LABELS[model]}",
                                line=dict(color=ACCENT, width=1.5)))
    st.plotly_chart(figure, width="stretch")
    st.write("Skala prognoz jest zwykle znacznie mniejsza niż skala rzeczywistych zwrotów: model uczony na zaszumionych "
             "danych ostrożnie trzyma się wartości bliskich średniej. Model, którego prognozy są prawie stałe, "
             "w praktyce kopiuje model naiwny.")
    glossary("Cel prognozy", "Model naiwny", "ARIMA", "LSTM", "Trafność kierunku", "MAE", "Odchylenie std prognoz", "Ziarno")


# --- Zakładka 3: Porównanie modeli -----------------------------------------------

def tab_comparison(metrics: pd.DataFrame, by_year: pd.DataFrame, by_anomaly: pd.DataFrame,
                   seed_metrics: pd.DataFrame | None) -> None:
    """Zakładka 3: tabela miar z testami, ziarna, trafność po latach, dni nietypowe (wszystko z plików results/)."""
    st.caption("Miary dla całego okresu testowego 2020–2025")
    st.write("Tabela zestawia wszystkie modele na dokładnie tych samych dniach testowych. "
             "Testy statystyczne liczone są tylko dla całego okresu, żeby uniknąć problemu wielokrotnego testowania.")
    table = metrics.copy()
    table["model"] = table["model"].map(MODEL_LABELS)
    show = table[["model", "n_days", "mae", "rmse", "direction_accuracy", "share_up_predictions", "pred_std",
                  "binomial_p_value", "mcnemar_p_value", "n_parameters", "fit_seconds_mean"]].rename(columns={
        "model": "Model", "n_days": "Dni", "mae": "MAE", "rmse": "RMSE", "direction_accuracy": "Trafność kierunku",
        "share_up_predictions": "Prognozy wzrostu", "pred_std": "Odch. std prognoz",
        "binomial_p_value": "p dwumianowy", "mcnemar_p_value": "p McNemara",
        "n_parameters": "Parametry", "fit_seconds_mean": "Czas uczenia okna [s]"})
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.5f", help=GLOSSARY["MAE"]),
            "RMSE": st.column_config.NumberColumn(format="%.5f", help=GLOSSARY["RMSE"]),
            "Trafność kierunku": st.column_config.NumberColumn(format="percent", help=GLOSSARY["Trafność kierunku"]),
            "Prognozy wzrostu": st.column_config.NumberColumn(format="percent", help=GLOSSARY["Odsetek prognoz wzrostu"]),
            "Odch. std prognoz": st.column_config.NumberColumn(format="%.5f", help=GLOSSARY["Odchylenie std prognoz"]),
            "p dwumianowy": st.column_config.NumberColumn(format="%.3f", help=GLOSSARY["Test dwumianowy"]),
            "p McNemara": st.column_config.NumberColumn(format="%.3f", help=GLOSSARY["Test McNemara"]),
            "Parametry": st.column_config.NumberColumn(help="Liczba parametrów modelu (wag), czyli jego złożoność."),
            "Czas uczenia okna [s]": st.column_config.NumberColumn(format="%.1f", help="Średni czas uczenia w jednym "
                                                                                      "oknie walk-forward (CPU, M1)."),
        },
    )
    st.write("Odczyt: żaden model nie trafia kierunku częściej niż model naiwny, a test McNemara nie wykazuje "
             "przewagi (p znacznie powyżej 0,05). Wartości p testu dwumianowego są małe dla wszystkich modeli, "
             "bo w latach 2020–2025 cena NVDA rosła w większości dni i każdy model przewidujący głównie wzrosty "
             "trafia częściej niż w 50% dni. Dla modelu naiwnego test McNemara wobec samego siebie nie ma sensu (p = 1).")

    if seed_metrics is not None and seed_metrics["seed"].nunique() > 1:
        st.caption("Trafność kierunku każdego ziarna LSTM")
        st.write("Każde ziarno to osobne uczenie tej samej sieci z innym startem losowym. Rozrzut między ziarnami "
                 "pokazuje, jaka część różnic między modelami to zwykły przypadek.")
        seeds = seed_metrics.pivot(index="model", columns="seed", values="direction_accuracy")
        seeds.index = seeds.index.map(MODEL_LABELS)
        seeds.columns = [f"Ziarno {c}" for c in seeds.columns]
        seeds.insert(0, "Model", seeds.index)
        st.dataframe(seeds, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in seeds.columns if c != "Model"})

    st.caption("Trafność kierunku w kolejnych latach (opisowo, bez wartości p)")
    figure = base_figure("Trafność kierunku", height=380)
    for model in [m for m in MODEL_LABELS if m in by_year["model"].unique()]:
        part = by_year[by_year["model"] == model].sort_values("year")
        figure.add_trace(go.Bar(x=part["year"].astype(str), y=part["direction_accuracy"], name=MODEL_LABELS[model],
                                marker_color=MODEL_COLORS.get(model, GRAY)))
    figure.update_layout(barmode="group", bargap=0.25)
    low, high = by_year["direction_accuracy"].min(), by_year["direction_accuracy"].max()
    figure.update_yaxes(tickformat=".0%", range=[min(0.4, low - 0.05), max(0.65, high + 0.05)])
    figure.add_hline(y=0.5, line=dict(color=LIGHT_GRAY, dash="dot"))
    st.plotly_chart(figure, width="stretch")
    st.write("Linia 50% to poziom rzutu monetą. Wysokość słupka modelu naiwnego to po prostu odsetek dni wzrostowych "
             "w danym roku (2022 był rokiem spadkowym, stąd poniżej 50%).")

    st.caption("Błędy w dniach nietypowych i zwykłych (opisowo)")
    anomaly_table = by_anomaly.copy()
    anomaly_table["model"] = anomaly_table["model"].map(MODEL_LABELS)
    anomaly_table["day_type"] = anomaly_table["day_type"].map({"anomalous": "nietypowe", "normal": "zwykłe"})
    anomaly_show = anomaly_table[["model", "day_type", "n_days", "mae", "rmse", "direction_accuracy"]].rename(columns={
        "model": "Model", "day_type": "Dni", "n_days": "Liczba dni", "mae": "MAE", "rmse": "RMSE",
        "direction_accuracy": "Trafność kierunku"})
    st.dataframe(anomaly_show, hide_index=True, width="stretch",
                 column_config={"MAE": st.column_config.NumberColumn(format="%.5f"),
                                "RMSE": st.column_config.NumberColumn(format="%.5f"),
                                "Trafność kierunku": st.column_config.NumberColumn(format="percent")})
    st.write("W dniach nietypowych błąd MAE każdego modelu jest prawie dwukrotnie większy niż w dniach zwykłych. "
             "Dni nietypowych jest niewiele, więc ich trafność kierunku ma charakter wyłącznie opisowy.")
    glossary("Trafność kierunku", "MAE", "RMSE", "Odchylenie std prognoz", "Test dwumianowy", "Test McNemara",
             "Wartość p", "Ziarno", "Dni nietypowe", "Walk-forward")


# --- Zakładka 4: Symulacja inwestowania --------------------------------------------

def tab_simulation(predictions: pd.DataFrame) -> None:
    """Zakładka 4: strategie przeliczane na żywo z zapisanych prognoz (jedyne obliczenia w aplikacji)."""
    st.caption("Wartość portfela strategii na prognozach wobec „kup i trzymaj”")
    st.write("Suwaki przeliczają strategie na żywo z zapisanych prognoz. Kapitał początkowy to 1,0, gotówka nie "
             "przynosi odsetek, bez krótkiej sprzedaży i dźwigni. Decyzja po zamknięciu dnia t dotyczy zwrotu dnia t+1.")
    columns = st.columns(3)
    models = [m for m in MODEL_LABELS if m in predictions["model"].unique()]
    model = columns[0].selectbox("Model", models, index=len(models) - 1, format_func=MODEL_LABELS.get,
                                 help="Model, którego prognozy sterują strategią podstawową i strategią z progiem.")
    commission_pct = columns[1].slider("Prowizja za transakcję [%]", 0.0, 0.5, 0.1, 0.05,
                                       help="Koszt każdego wejścia i wyjścia z pozycji jako procent wartości portfela. "
                                            "Płaci ją także „kup i trzymaj” (raz przy kupnie, raz przy sprzedaży).")
    threshold_pct = columns[2].slider("Próg prognozy dla strategii z progiem [%]", 0.0, 1.0, 0.1, 0.05,
                                      help=GLOSSARY["Strategia z progiem"])

    metrics, curves = evaluate_strategies(predictions, commission=commission_pct / 100, threshold=threshold_pct / 100)
    rows = {
        "Kup i trzymaj": metrics[metrics["strategy"] == "buy_and_hold"].iloc[0],
        "Podstawowa": metrics[(metrics["model"] == model) & (metrics["strategy"] == "basic")].iloc[0],
        "Z progiem": metrics[(metrics["model"] == model) & (metrics["strategy"] == "threshold")].iloc[0],
    }
    curve_names = {"Kup i trzymaj": "buy_and_hold", "Podstawowa": f"{model}__basic", "Z progiem": f"{model}__threshold"}
    terms = {"Kup i trzymaj": "Kup i trzymaj", "Podstawowa": "Strategia podstawowa", "Z progiem": "Strategia z progiem"}

    for name, row in rows.items():
        st.caption(name if name == "Kup i trzymaj" else f"{name} · {MODEL_LABELS[model]}")
        cols = st.columns(6)
        metric(cols[0], "Zwrot całkowity", percent(row["total_return"]))
        metric(cols[1], "CAGR", percent(row["cagr"]))
        metric(cols[2], "Maks. obsunięcie", percent(row["max_drawdown"]), "Maksymalne obsunięcie")
        metric(cols[3], "Sharpe", "–" if np.isnan(row["sharpe"]) else f"{row['sharpe']:.2f}", "Wskaźnik Sharpe'a")
        metric(cols[4], "Transakcje", f"{int(row['n_trades'])}")
        metric(cols[5], "Dni w akcjach", percent(row["share_days_in_stock"]))

    figure = base_figure("Wartość portfela")
    styles = {"Kup i trzymaj": dict(color=LIGHT_GRAY, width=1.5), "Podstawowa": dict(color=ACCENT, width=1.5),
              "Z progiem": dict(color=DARK_GRAY, width=1.5, dash="dot")}
    for name, column in curve_names.items():
        figure.add_trace(go.Scatter(x=curves.index, y=curves[column], mode="lines", name=name, line=styles[name]))
    figure.update_yaxes(type="log", title="Wartość portfela (skala logarytmiczna)")
    st.plotly_chart(figure, width="stretch")
    st.write("Skala logarytmiczna: równe odstępy pionowe oznaczają równe procentowe zmiany wartości. Wartość przy dacie t uwzględnia już zwrot dnia t+1, o którym decyzja zapadła w dniu t. "
             "Model naiwny ze strategią podstawową daje dokładnie krzywą „kup i trzymaj”. Jeśli prognozy są bliskie zera, "
             "strategia z progiem może prawie cały czas trzymać gotówkę; to wynik do opisania, nie usterka.")
    glossary("Kup i trzymaj", "Strategia podstawowa", "Strategia z progiem", "Zwrot całkowity", "CAGR",
             "Maksymalne obsunięcie", "Wskaźnik Sharpe'a", "Transakcje", "Dni w akcjach")


# --- Zakładka 5: O projekcie -----------------------------------------------------

def tab_about(run_info: dict | None) -> None:
    """Zakładka 5: cel, metoda, ograniczenia, ustawienia użyte w eksperymencie, zastrzeżenie."""
    st.caption("Cel, metoda i ograniczenia")
    st.markdown(
        """
**Cel.** Sprawdzić, czy sieć LSTM prognozuje dzienny zwrot i kierunek akcji NVIDIA na następny dzień lepiej niż
model naiwny (prognoza 0, czyli „jutro bez zmian”) i ARIMA, oraz czy wskaźniki techniczne poprawiają wynik LSTM.
Wynik negatywny jest w pełni akceptowalny i pokazany uczciwie.

**Dane.** Dzienne notowania NVDA i QQQ (fundusz odwzorowujący NASDAQ-100, użyty tylko jako cecha „co robił rynek”)
z serwisu Yahoo Finance, ceny skorygowane o podziały i dywidendy, lata 2015–2025.
Cel prognozy: zwrot logarytmiczny `ln(Close(t+1) / Close(t))`, prognozowany po zamknięciu dnia t wyłącznie z danych do dnia t.

**Modele.** Naiwny (0), ARIMA na zwrotach (rząd po AIC na 2015–2019), LSTM A (zwrot i zmiana wolumenu),
LSTM B (jak A oraz Close/SMA20, Close/SMA50, RSI 14, zmienność 20 dni, zwrot QQQ). LSTM uczony 5 razy (ziarna 0–4).

**Schemat walk-forward.** Okres testowy 2020–2025 podzielony na 24 kwartały. Na początku każdego kwartału model uczy się
na wszystkich danych od 2015 r. do dnia przed kwartałem (okno rozszerzające), potem prognozuje każdy dzień kwartału.
Skalowanie i Isolation Forest są dopasowywane tylko na danych uczących okna. Ustawienia LSTM strojone raz
(nauka 2015–2018, ocena 2019) i potem zamrożone.

```
2015 ──────────── 2019 | 2020Q1 | 2020Q2 | ... | 2025Q4
[ uczenie            ] [ test  ]
[ uczenie                     ] [ test  ]
[ uczenie                              ] [ test  ]  ... 24 okna
```

**Miary.** MAE, RMSE, trafność kierunku, odsetek prognoz wzrostu i odchylenie standardowe prognoz (kontrola
„zapadnięcia” do stałej), dokładny test McNemara wobec modelu naiwnego i test dwumianowy wobec 50%.

**Ograniczenia.** Jedna spółka; gotówka bez odsetek; stała prowizja bez poślizgu cenowego; transakcje po cenie zamknięcia;
dane z jednego dostawcy; brak danych fundamentalnych i wiadomości; wyniki zależne od okresu 2020–2025.

**To nie jest porada inwestycyjna.** Aplikacja ma charakter edukacyjno-badawczy.
        """
    )
    if run_info:
        st.caption("Ustawienia użyte w eksperymencie")
        chosen = run_info.get("chosen_config", {})
        order = run_info.get("arima_order")
        lines = []
        if order:
            lines.append(f"Rząd ARIMA (p, d, q) wybrany kryterium AIC: ({order[0]}, {order[1]}, {order[2]}).")
        for variant in ("A", "B"):
            settings = chosen.get(variant)
            if settings:
                lines.append(f"LSTM {variant}: okno {settings['window']} dni, {settings['units']} neuronów, "
                             f"{settings['layers']} {'warstwa' if settings['layers'] == 1 else 'warstwy'} LSTM.")
        lines.append(f"Liczba dni testowych: {run_info.get('n_test_days')}, okien walk-forward: {run_info.get('n_windows')}.")
        st.write(" ".join(lines))
        libraries = run_info.get("libraries", {})
        if libraries:
            st.write("Wersje bibliotek: " + ", ".join(f"{k} {v}" for k, v in libraries.items()) + ".")
    glossary("Cel prognozy", "Walk-forward", "Model naiwny", "ARIMA", "LSTM", "Dni nietypowe", "Test McNemara")
    st.caption("Praca inżynierska, WSB Merito. Zespół: Konrad Kowalewski, Stanisław Kucia, Kacper Langa.")


# --- Główna funkcja -------------------------------------------------------------

def main() -> None:
    """Układ strony: tytuł, informacja o trybie wyników, pięć zakładek."""
    st.set_page_config(page_title="Prognozowanie zwrotów NVDA siecią LSTM", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Prognozowanie dziennych zwrotów NVIDIA siecią LSTM")
    st.caption("Praca inżynierska · porównanie LSTM z modelem naiwnym i ARIMA · walk-forward 2020–2025")

    prices = load_prices()
    if not results_available():
        st.warning("Brak plików wyników w katalogu results/. Uruchom `python -m src.run_experiment --quick`, żeby je wygenerować.")
        tab_data(prices, None, None)
        return

    run_info = load_json("run_info.json")
    # Spec: aplikacja mówi, z którego trybu pochodzą wyniki (np. demo = 1 powtórzenie).
    st.write(run_info.get("mode_description", f"Tryb: {run_info.get('mode')}") + ".")

    predictions = load_csv("predictions.csv")
    anomalies = load_csv("anomalies.csv")
    descriptive_path = RESULTS_DIR / "anomalies_descriptive_2015_2019.csv"
    descriptive = load_csv(descriptive_path.name) if descriptive_path.exists() else None
    metrics = load_csv("metrics.csv")
    by_year = load_csv("metrics_by_year.csv")
    by_anomaly = load_csv("metrics_anomalies.csv")
    seed_metrics = load_csv("metrics_seeds.csv") if (RESULTS_DIR / "metrics_seeds.csv").exists() else None

    tabs = st.tabs(["Dane", "Prognozy", "Porównanie modeli", "Symulacja inwestowania", "O projekcie"])
    with tabs[0]:
        tab_data(prices, anomalies, descriptive)
    with tabs[1]:
        tab_predictions(predictions)
    with tabs[2]:
        tab_comparison(metrics, by_year, by_anomaly, seed_metrics)
    with tabs[3]:
        tab_simulation(predictions)
    with tabs[4]:
        tab_about(run_info)


main()
