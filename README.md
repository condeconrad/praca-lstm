# Prognozowanie dziennych zwrotów akcji NVIDIA siecią LSTM

Projekt inżynierski, kierunek Informatyka, Uniwersytet WSB Merito.
Zespół: Konrad Kowalewski, Stanisław Kucia, Kacper Langa. Promotor: dr Michał Stasiak.

System prognozuje dzienny zwrot logarytmiczny akcji NVIDIA (NVDA) na następny dzień siecią LSTM i porównuje ją
z modelem naiwnym (prognoza zerowego zwrotu) oraz modelem ARIMA. Ocena odbywa się w schemacie walk-forward:
24 kwartały 2020–2025, ponowne uczenie na początku każdego kwartału, wszystkie modele na tych samych 1508 dniach
testowych. Wynik uzupełniają testy statystyczne, analiza dni nietypowych (Isolation Forest) i symulacja strategii
inwestycyjnych z prowizją. Wyniki przegląda aplikacja internetowa w Streamlit.

**Pytania badawcze**

- P1: Czy LSTM prognozuje zwrot i kierunek NVDA na następny dzień lepiej niż model naiwny i ARIMA?
- P2: Czy wskaźniki techniczne poprawiają wynik LSTM (wariant A wobec wariantu B)?
- P3: Czy strategia oparta na prognozach daje po prowizjach lepszy wynik niż "kup i trzymaj"?
- P4: Czy modele mylą się bardziej po dniach nietypowych?

## Najważniejsze wyniki

Pełny przebieg: 24 kwartały, LSTM uczony 5 razy (ziarna 0–4), prognozy uśrednione. Źródło: `results/metrics.csv`.

| Model | MAE | Trafność kierunku | p testu McNemara wobec modelu naiwnego |
|---|---|---|---|
| Naiwny (prognoza 0) | 0,02451 | 54,6% | – |
| ARIMA (rząd 0,0,0 wybrany kryterium AIC) | 0,02438 | 54,6% | 1,000 |
| LSTM A (zwrot, wolumen) | 0,02435 | 54,2% | 0,706 |
| LSTM B (A + wskaźniki techniczne + QQQ) | 0,02442 | 53,8% | 0,899 |

- **P1, P2:** nie. Żaden model nie trafia kierunku częściej niż model naiwny, a wskaźniki techniczne nie pomagają.
- **P3:** nie. Strategia "kup i trzymaj" (+3056%) wygrywa ze wszystkimi strategiami na prognozach (`results/strategies.csv`).
- **P4:** tak. Po dniach nietypowych błąd MAE każdego modelu jest prawie dwukrotnie większy (`results/metrics_anomalies.csv`).

Wynik negatywny jest zgodny ze słabą formą hipotezy rynku efektywnego. Wartością projektu jest metoda oceny,
która pozwala taką odpowiedź sformułować wiarygodnie.

## Aplikacja

Aplikacja ma pięć zakładek: Dane, Prognozy, Porównanie modeli, Symulacja inwestowania, O projekcie.
Tylko czyta pliki z `data/raw/` i `results/`, niczego nie trenuje, więc uruchamia się w kilka sekund i nie wymaga
TensorFlow. Każda liczba ma dymek z objaśnieniem, a każda zakładka słowniczek pojęć. Suwaki prowizji i progu
przeliczają strategie na żywo.

Uruchomienie lokalne (po instalacji opisanej niżej):

```bash
.venv/bin/streamlit run app/app.py
```

Aplikacja otworzy się pod adresem http://localhost:8501.

Do samej aplikacji wystarczy lżejsze środowisko:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-app.txt
.venv/bin/streamlit run app/app.py
```

Wersja w przeglądarce bez instalacji: repozytorium jest przygotowane do Streamlit Community Cloud
(plik główny `app/app.py`, zależności `app/requirements.txt`, motyw `.streamlit/config.toml`, Python 3.12).

## Instalacja pełnego środowiska (macOS, Apple Silicon, CPU)

```bash
git clone https://github.com/condeconrad/praca-lstm.git
cd praca-lstm
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Polecenia uruchamiamy przez `.venv/bin/python`, `.venv/bin/pytest` i `.venv/bin/streamlit`. Wersje bibliotek są
przypięte w `requirements.txt`; wersje użyte w przebiegu zapisuje `results/run_info.json`.

## Trening modeli i odtworzenie wyników

Wyniki i dane są w repozytorium, więc aplikacja działa od razu. Cały eksperyment, łącznie z uczeniem wszystkich
modeli, odtwarza jedno polecenie:

```bash
.venv/bin/python -m src.run_experiment --quick   # 2 kwartały, 1 ziarno, 5 epok: sprawdzenie działania (ok. 1 min)
.venv/bin/python -m src.run_experiment --demo    # 24 kwartały, ziarno 0 (ok. 8 min)
.venv/bin/python -m src.run_experiment           # pełny: 24 kwartały, ziarna 0–4, wyniki z tabeli wyżej (ok. 28 min)
```

Kolejność kroków: dane, tabela cech, okna walk-forward, wybór rzędu ARIMA (AIC, lata 2015–2019), strojenie LSTM
(uczenie 2015–2018, ocena 2019), uczenie i prognozy w 24 oknach z zapisem modeli, Isolation Forest, miary, strategie.
Każdy tryb zapisuje te same pliki do `results/`; `results/run_info.json` podaje tryb, czas trwania i wersje bibliotek,
a `results/full_log.txt` to zapis przebiegu pełnego treningu. Ziarna losowości i deterministyczne operacje
TensorFlow są ustawione, więc powtórzony przebieg na tej samej maszynie daje identyczne prognozy.

Dane pochodzą z Yahoo Finance (biblioteka yfinance) i są w `data/raw/` (NVDA i QQQ, 2776 sesji, 2015-01-02 – 2026-01-15,
ceny skorygowane o podziały i dywidendy). Polecenie `.venv/bin/python -m src.data --refresh` pobiera je ponownie.
Pliki kontrolne ze Stooq można dodać do `data/control/`; wtedy powstaje raport różnic `results/data_check.csv`.

## Wytrenowane modele

Każdy przebieg zapisuje modele do `models/`:

```
models/chosen_config.json                               wybrane ustawienia LSTM ze strojenia
models/arima/<kwartał>.json                             parametry ARIMA każdego okna
models/lstm/<A|B>/<kwartał>_seed<n>.keras               sieć Keras (architektura i wagi)
models/lstm/<A|B>/<kwartał>_seed<n>_scalers.joblib      skalery, lista cech, ustawienia, ostatnie dni uczące
```

W repozytorium są parametry ARIMA wszystkich okien i modele LSTM ostatniego okna (2025Q4, oba warianty, ziarna 0–4).
Pozostałe modele LSTM (ok. 52 MB) powstają lokalnie przy uruchomieniu eksperymentu.

Wczytanie modelu i odtworzenie prognoz bez ponownego uczenia:

```python
from src import config
from src.data import ensure_data
from src.features import build_feature_table
from src.models import ArimaModel, LstmModel
from src.walk_forward import make_windows

prices = ensure_data()
features = build_feature_table(prices["NVDA"], prices["QQQ"])
window = [w for w in make_windows(features) if w.name == "2025Q4"][0]

lstm = LstmModel.load(config.MODELS_DIR / "lstm" / "A" / "2025Q4_seed0")
y_pred_lstm = lstm.predict(window.test)

arima = ArimaModel.load(config.MODELS_DIR / "arima" / "2025Q4.json", window.train)
y_pred_arima = arima.predict(window.test)
```

Prognozy są identyczne z zapisanymi w `results/predictions_seeds.csv`; sprawdza to `tests/test_model_saving.py`.

## Testy

```bash
.venv/bin/pytest -q
```

40 testów sprawdza m.in.: brak wycieku danych z przyszłości, wyrównanie celu (zwrot z dnia t na t+1), chronologię
i rozłączność okien walk-forward, identyczne dni testowe dla wszystkich modeli, skaler dopasowany tylko na danych
uczących, miary i prowizje na przykładach liczonych ręcznie, równość "model naiwny + strategia podstawowa = kup
i trzymaj", zapis i wczytanie modeli oraz uruchomienie aplikacji.

## Struktura repozytorium

```
src/                    kod eksperymentu
  config.py             wszystkie parametry
  data.py               pobieranie, zapis i walidacja danych
  features.py           zwroty, cel prognozy, wskaźniki techniczne
  models.py             BaseModel, NaiveModel, ArimaModel, LstmModel (zapis i wczytanie)
  walk_forward.py       okna kwartalne i pętla uczenia
  tuning.py             strojenie LSTM
  anomalies.py          Isolation Forest
  metrics.py            MAE, RMSE, trafność kierunku, test McNemara i dwumianowy
  strategy.py           symulacja strategii z prowizją
  run_experiment.py     cały eksperyment, zapis results/ i models/
app/app.py              aplikacja Streamlit
tests/                  testy pytest
data/raw/               dane NVDA i QQQ z metadanymi
results/                wyniki (CSV, JSON) i logi przebiegów
models/                 wytrenowane modele
docs/                   specyfikacja, architektura z diagramem, dziennik decyzji, pomysły, źródła do weryfikacji
```

## Dokumentacja

- `docs/SPECYFIKACJA.md`: założenia, pytania badawcze i pełna specyfikacja.
- `docs/ARCHITEKTURA.md`: moduły, przepływ danych (diagram Mermaid), pliki wyników.
- `docs/DZIENNIK.md`: dziennik decyzji technicznych z uzasadnieniami.
- `docs/pomysly.md`: kierunki rozwoju poza zakresem pracy.

## Ograniczenia

Jedna spółka i okres 2020–2025; gotówka bez odsetek; stała prowizja bez poślizgu cenowego; transakcje po cenie
zamknięcia; dane z jednego dostawcy; brak danych fundamentalnych i wiadomości.

Projekt ma charakter edukacyjno-badawczy i nie jest poradą inwestycyjną.
