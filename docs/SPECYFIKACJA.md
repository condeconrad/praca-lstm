# Specyfikacja projektu

Prognozowanie dziennych zwrotów akcji NVIDIA siecią LSTM. Praca inżynierska (informatyka, WSB Merito), zespół:
Konrad Kowalewski, Stanisław Kucia, Kacper Langa. Ten plik jest jedynym źródłem prawdy o założeniach projektu;
zmiany założeń wpisujemy tutaj i odnotowujemy w `docs/DZIENNIK.md`.

## Cel projektu

Zaprojektować, zbudować i rzetelnie przetestować system, który prognozuje dzienny zwrot akcji NVIDIA (NVDA)
siecią neuronową LSTM i porównuje ją z prostymi modelami odniesienia. Wynik negatywny
(LSTM nie lepszy od modelu naiwnego) jest w pełni akceptowalny i ma zostać pokazany uczciwie.

Pytania badawcze:
- P1: Czy LSTM prognozuje zwrot i kierunek NVDA na następny dzień lepiej niż model naiwny i ARIMA?
- P2: Czy wskaźniki techniczne poprawiają wynik LSTM (LSTM A vs LSTM B)?
- P3: Czy strategia oparta na prognozach daje po prowizjach lepszy wynik niż "kup i trzymaj"?
- P4: Czy modele mylą się bardziej w dniach nietypowych (wskazanych przez Isolation Forest)?

## Zasady projektu

1. **Każda liczba pochodzi z uruchomienia kodu.** Metryki, wyniki i czasy w dokumentacji i w pracy pochodzą z plików
   w katalogu `results/` (z podaniem pliku źródłowego). Czego nie uruchomiono, tego nie opisujemy jako wyniku.
2. **API bibliotek sprawdzamy w zainstalowanej wersji** (`help()`, `dir()`, `pip show`), a nie z pamięci.
3. **Źródła tylko zweryfikowane.** Bibliografia pracy korzysta wyłącznie ze źródeł z sekcji "Źródła zweryfikowane"
   na końcu tego pliku. Propozycje nowych źródeł trafiają do `docs/zrodla_do_weryfikacji.md` (autor, tytuł, rok, link lub DOI,
   do czego potrzebne) i nie są używane, dopóki zespół ich nie przeczyta. Bez dosłownych cytatów i numerów stron,
   których nikt nie sprawdził.
4. **Zero wycieku danych z przyszłości.** Prognoza na dzień t+1 może korzystać wyłącznie z danych do dnia t włącznie.
   Skalowanie, Isolation Forest i dobór parametrów dopasowujemy tylko na danych uczących danego okna.
5. **Zakres jest zamknięty.** Robimy dokładnie to, co jest w tym pliku; pomysły na rozszerzenia zapisujemy w `docs/pomysly.md`.
6. **Prostota ponad spryt.** Kod ma być zrozumiały dla przeciętnego studenta informatyki znającego C#,
   a słabiej Pythona: czytelne nazwy, krótkie funkcje, bez zaawansowanych sztuczek językowych.
7. **Testy po każdym etapie** (`pytest`); nie przechodzimy dalej, jeśli testy nie przechodzą.
8. **Commit po każdym etapie** z opisowym komunikatem po polsku. Nigdy nie commitujemy `.venv/`.

## Konwencje

- Python 3.12, środowisko wirtualne `.venv`, sprzęt: MacBook M1, obliczenia na CPU (bez GPU, bez tensorflow-metal).
- Nazwy w kodzie (zmienne, funkcje, klasy, pliki) po angielsku. Komentarze, docstringi, README, dokumentacja
  i cały interfejs aplikacji po polsku.
- Wszystkie parametry w jednym miejscu: `src/config.py` (zwykły moduł Pythona, bez YAML).
- Biblioteki: pandas, numpy, yfinance, statsmodels, scikit-learn, scipy, tensorflow (Keras), streamlit, plotly, pytest.
  Wskaźniki techniczne licz samodzielnie w pandas (bez dodatkowych bibliotek do analizy technicznej).
- Ustawiaj ziarna losowości (random, numpy, tensorflow) dla powtarzalności; sprawdź w zainstalowanej wersji
  TensorFlow, czy dostępne jest wymuszenie deterministycznych operacji, i opisz wynik w `docs/DZIENNIK.md`.

## Struktura repozytorium

```
README.md
requirements.txt          # pełne środowisko (eksperyment + aplikacja + testy)
requirements-app.txt      # tylko to, czego potrzebuje aplikacja (streamlit, pandas, plotly)
.gitignore
src/
  config.py               # wszystkie parametry
  data.py                 # pobieranie, zapis i walidacja danych
  features.py             # zwroty, cel prognozy, wskaźniki
  models.py               # BaseModel, NaiveModel, ArimaModel, LstmModel
  walk_forward.py         # podział na okna i pętla walk-forward
  anomalies.py            # Isolation Forest
  metrics.py              # MAE, RMSE, trafność kierunku, test McNemara i dwumianowy
  strategy.py             # symulacja strategii z prowizją
  tuning.py               # strojenie ustawień LSTM
  run_experiment.py       # uruchamia cały eksperyment, zapisuje results/
app/
  app.py                  # aplikacja Streamlit (tylko czyta data/ i results/)
tests/
data/
  raw/                    # NVDA.csv, QQQ.csv, metadata.json
  control/                # opcjonalne pliki kontrolne ze Stooq (dodaje zespół ręcznie)
results/                  # wszystkie wyniki w CSV/JSON
models/                   # wytrenowane modele z walk-forward (LSTM .keras + skalery, ARIMA .json)
docs/
  SPECYFIKACJA.md         # ten plik
  ARCHITEKTURA.md         # opis modułów + diagram (Mermaid)
  DZIENNIK.md             # dziennik decyzji: data, decyzja, powód
  pomysly.md
  zrodla_do_weryfikacji.md  # propozycje źródeł, nieużywane do czasu potwierdzenia przez zespół
```

## Specyfikacja

### Dane
- Instrumenty: **NVDA** (jedyny obiekt badania, decyzja zespołu i promotora) i **QQQ** (wyłącznie jako cecha wejściowa
  "co robił rynek"; nie jest osobno prognozowany ani oceniany). Kod przyjmuje listę tickerów w `config.py`,
  ale w tej pracy lista to `["NVDA"]`; rozszerzenie na QQQ jako drugi obiekt jest tylko pomysłem w `docs/pomysly.md`.
- Okres badania: 2015-01-01 do 2025-12-31, notowania dzienne, źródło yfinance, ceny skorygowane o podziały i dywidendy
  (`auto_adjust=True`; sprawdź w zainstalowanej wersji, jak to się nazywa).
- Pobierz dane kilka sesji dalej (do połowy stycznia 2026), żeby ostatnia sesja 2025 miała znany zwrot następnego dnia.
  Sprawdź, czy parametr końca zakresu w yfinance jest włączny czy wyłączny. Dni po 2025-12-31 nie są dniami testowymi.
- Zapis do `data/raw/NVDA.csv` i `data/raw/QQQ.csv` oraz `data/raw/metadata.json`
  (wersja yfinance, liczba wierszy, zakres dat). Jeśli pliki istnieją, nie pobieraj ponownie
  (chyba że podano flagę `--refresh`).
- Walidacja: brak duplikatów dat, daty rosnące, ceny dodatnie, raport brakujących wartości.
  Weekendy i święta giełdowe to normalne braki, niczego nie uzupełniaj.
  Jeśli ostatnia sesja w danych jest wcześniejsza niż 2026-01-02, przerwij z czytelnym komunikatem.
- Kontrola jakości: jeśli w `data/control/` są pliki CSV ze Stooq (nagłówki mogą być po polsku lub po angielsku),
  porównaj ceny zamknięcia dla wspólnych dat i zapisz raport różnic do `results/data_check.csv`.
  Pamiętaj, że różnice mogą wynikać z odmiennej korekty o dywidendy; raportuj, nie poprawiaj.

### Cel prognozy
- Zwrot logarytmiczny na następny dzień: `y_t = ln(Close_{t+1} / Close_t)`, znany w chwili zamknięcia dnia t+1.
- Prognoza powstaje po zamknięciu dnia t, wyłącznie z danych do dnia t.
- Kierunek: wzrost, gdy zwrot >= 0; w przeciwnym razie spadek. Ta sama reguła dla prognoz i dla wartości rzeczywistych.

### Cechy wejściowe
- **Wariant A (LSTM A):** dzienny zwrot logarytmiczny NVDA, zmiana logarytmu wolumenu NVDA.
- **Wariant B (LSTM B):** jak A oraz:
  - `Close / SMA20 - 1` i `Close / SMA50 - 1`,
  - RSI z 14 dni (metoda Wildera),
  - zmienność: odchylenie standardowe zwrotów z 20 dni,
  - dzienny zwrot logarytmiczny QQQ.
- Każda cecha w dniu t liczona tylko z danych do dnia t.
- **Wspólny zbiór dni:** cechy liczone na całym szeregu (korzystanie z dni wcześniejszych jest dozwolone), wiersze z brakami
  (rozbieg SMA50, RSI, zmienności) odrzucane raz, na początku. Wszystkie modele oceniane dokładnie na tych samych
  dniach testowych. Okno LSTM na początku kwartału sięga do dni z okresu uczącego (dane przeszłe). Test pytest sprawdza,
  że lista dat w `predictions.csv` jest identyczna dla każdego modelu.

### Modele (wspólny interfejs)
Klasa bazowa `BaseModel` z metodami `fit(train_data)` i `predict(test_data)`, zwracającą prognozowany zwrot dla każdego dnia.

1. **NaiveModel:** prognoza zwrotu = 0 (a więc kierunek: zawsze wzrost).
2. **ArimaModel:** ARIMA (statsmodels) na zwrotach logarytmicznych, d = 0.
   Rząd (p, q) dla p, q ∈ {0, 1, 2} wybrany kryterium AIC na latach 2015–2019, potem stały.
   Parametry ponownie estymowane na początku każdego kwartału testowego; w trakcie kwartału prognozy
   jednodniowe z aktualizacją stanu o nowe obserwacje bez ponownej estymacji
   (sprawdź w zainstalowanej wersji statsmodels, jak to zrobić poprawnie).
   Plan awaryjny, równie poprawny: ponowna estymacja ARIMA każdego dnia w pętli (tanie obliczeniowo).
3. **LstmModel (Keras):**
   - Architektura: wejście (okno × liczba cech) → 1 lub 2 warstwy LSTM → Dropout 0,2 → Dense(1).
   - Uczenie: optymalizator Adam (domyślne tempo uczenia), funkcja straty MAE, batch 32,
     maksymalnie 50 epok, EarlyStopping(patience=5, restore_best_weights=True).
   - Walidacja wewnętrzna: **ostatnie 20% danych uczących chronologicznie** (nigdy losowo).
   - Skalowanie: StandardScaler dla cech i celu, dopasowany tylko na części uczącej (bez walidacji); po prognozie odwróć skalowanie celu.
   - 5 powtórzeń z ziarnami 0–4, prognozy uśredniane. Prognozy każdego ziarna zapisywane osobno
     (`results/predictions_seeds.csv`), a trafność kierunku raportowana dla każdego ziarna oraz dla średniej (min, max, średnia).
   - Świadoma decyzja: ostatnie 20% danych uczących służy tylko do wczesnego zatrzymania, więc model nie uczy się
     na najświeższych danych okna. Akceptujemy to dla prostoty; alternatywa (ponowny trening na całości) nie wchodzi w zakres.

### Strojenie LSTM (tylko raz, przed testem)
- Siatka: okno {20, 60} × neurony {32, 64} × warstwy {1, 2} = 8 kombinacji, osobno dla wariantu A i B.
- Nauka na 2015–2018, ocena na 2019, 3 powtórzenia na kombinację. Zapisz dla każdej kombinacji: MAE, trafność kierunku,
  odchylenie standardowe prognoz.
- Kryterium wyboru: najniższe MAE, ale z regułą odrzucenia modeli "zapadniętych": kombinacja, której odchylenie
  standardowe prognoz na 2019 jest mniejsze niż 20% najwyższego odchylenia standardowego wśród kombinacji, odpada.
  Powód: na zaszumionych zwrotach najniższe MAE osiąga sieć prognozująca prawie stałe zero, czyli kopia modelu naiwnego.
  Jeśli wszystkie kombinacje odpadają, wybierz najwyższe odchylenie standardowe i zapisz to w `DZIENNIK.md`.
- Wyniki do `results/tuning.csv`, wybrane ustawienia zapisz w `results/chosen_config.json` i użyj ich w teście.

### Test walk-forward
- Okres testowy: 2020-Q1 do 2025-Q4, czyli 24 kwartały.
- Okno rozszerzające: na początku każdego kwartału model uczy się na wszystkich danych od 2015-01-01
  do ostatniego dnia przed kwartałem, potem prognozuje każdy dzień kwartału.
- Te same okna dla wszystkich modeli.
- Wszystkie prognozy do `results/predictions.csv`: data, model, y_true, y_pred.
- Tryby uruchomienia (`run_experiment.py`):
  - `--quick`: 2 kwartały, 1 powtórzenie, 5 epok; sprawdzenie, że całość działa (minuty).
  - `--demo`: pełne 24 kwartały, 1 powtórzenie (ziarno 0), pełne epoki; wyniki do pokazania promotorowi (ok. 1–2 h).
  - pełny (bez flagi): 24 kwartały, 5 powtórzeń; wyniki do pracy (noc).
  Aplikacja i pliki wyników mają identyczny format w każdym trybie; `run_info.json` zapisuje, który tryb je wygenerował,
  a aplikacja wyświetla to w nagłówku ("Wyniki z trybu demo: 1 powtórzenie").

### Isolation Forest (anomalie)
- Cechy: wartość bezwzględna dziennego zwrotu, zmiana logarytmu wolumenu, zmienność z 20 dni.
- `contamination = 0.02`, `random_state = 42`. Próg wyznaczany na danych uczących, więc odsetek dni nietypowych
  w danych testowych może być inny niż 2% (np. większy w 2020 r.); to poprawne i trzeba to wyjaśnić w pracy.
- Ocena: w każdym oknie walk-forward dopasuj na danych uczących, oznacz dni testowe.
- Do wykresu opisowego lat 2015–2019 możesz dopasować model na tych latach, ale oznacz to jako analizę opisową,
  nieużywaną w ocenie.
- Wynik do `results/anomalies.csv`: data, czy_nietypowy, wynik_anomalii.

### Miary oceny
- MAE, RMSE, trafność kierunku (odsetek dni ze zgodnym kierunkiem).
- Odsetek prognoz wzrostu i odchylenie standardowe prognoz: pokazują, czy model nie "zapadł się"
  do prognozowania stałej bliskiej zera (wtedy w praktyce kopiuje model naiwny). To trzeba raportować wprost.
- Porównanie kierunku z modelem naiwnym na tych samych dniach: dokładny test McNemara,
  czyli `scipy.stats.binomtest` na dniach, w których dokładnie jeden z dwóch modeli trafił (p = 0,5), jednostronnie.
  Dodatkowo test dwumianowy trafności modelu względem 0,5. Poziom istotności 0,05.
- Testy statystyczne (McNemar, dwumianowy) tylko dla całego okresu 2020–2025, po jednym na model. Rozbicie roczne
  i dni nietypowe vs zwykłe raportowane opisowo, bez wartości p (unikamy problemu wielokrotnego testowania).
- Rozbicie: cały okres, każdy rok osobno, dni nietypowe vs zwykłe.
- Koszt modeli: liczba parametrów, czas uczenia jednego okna, czas całości – do `metrics.csv` i `run_info.json`.
- Pliki: `results/metrics.csv`, `results/metrics_by_year.csv`, `results/metrics_anomalies.csv`.

### Strategie inwestycyjne
- Decyzja po zamknięciu dnia t dotyczy zwrotu z dnia t+1. Zwroty strategii licz jako zwroty proste (`exp(y) - 1`).
- **Kup i trzymaj:** kupno na początku okresu testowego, trzymanie do końca.
- Każda strategia (także kup i trzymaj) płaci prowizję za wejście w pierwszym dniu i za wyjście w ostatnim dniu okresu
  (zamknięcie pozycji do wyceny). Dzięki temu równość "naiwny + podstawowa = kup i trzymaj" jest ścisła.
- **Podstawowa:** pozycja w akcjach, gdy prognoza >= 0; w przeciwnym razie gotówka.
- **Z progiem:** pozycja w akcjach, gdy prognoza > 0,001 (0,1%); w przeciwnym razie gotówka.
- Prowizja 0,1% przy każdym wejściu i wyjściu z pozycji. Gotówka nie przynosi odsetek. Bez krótkiej sprzedaży i dźwigni.
- Model naiwny ze strategią podstawową musi dawać ten sam wynik co "kup i trzymaj" (test to sprawdza).
- Miary: zwrot całkowity, CAGR = (wartość końcowa / początkowa)^(252 / liczba dni) − 1, maksymalne obsunięcie,
  wskaźnik Sharpe'a (roczny, stopa wolna od ryzyka 0, √252), liczba transakcji, odsetek dni w akcjach.
- Jeśli LSTM prognozuje wartości bardzo bliskie zera, strategia z progiem może być prawie cały czas w gotówce.
  To wynik do opisania, nie usterka.
- Pliki: `results/strategies.csv` (miary), `results/equity_curves.csv` (wartość portfela dzień po dniu).

### Informacje o przebiegu
- `results/run_info.json`: tryb, wersje Pythona i bibliotek, czas trwania, użyte parametry.

### Testy (pytest), minimum
- Brak wycieku: zmiana danych po dniu t nie zmienia cech dnia t.
- Poprawne wyrównanie celu: `y_t` to zwrot z t na t+1.
- Okna walk-forward: chronologiczne, dane testowe nie nachodzą na uczące.
- Identyczna lista dat testowych dla wszystkich modeli.
- Skaler dopasowany tylko na danych uczących.
- Miary na prostych danych z ręcznie policzonym wynikiem.
- Strategia: prowizja liczona poprawnie na prostym przykładzie; model naiwny = "kup i trzymaj".

### Aplikacja (Streamlit, po polsku)
Aplikacja tylko czyta pliki z `data/` i `results/`; niczego nie trenuje. Pięć zakładek:
(Styl wizualny: sekcja "Wygląd aplikacji" niżej.)
1. **Dane:** wykres ceny NVDA z wyborem okresu, zaznaczone dni nietypowe, podstawowe statystyki zwrotów.
2. **Prognozy:** wybór modelu, prognozy na tle rzeczywistych zwrotów, trafność kierunku.
3. **Porównanie modeli:** tabela miar, trafność kierunku w kolejnych latach, wynik testu dwumianowego, błędy w dniach nietypowych.
4. **Symulacja inwestowania:** wartość portfela strategii vs "kup i trzymaj"; suwaki prowizji i progu
   przeliczają strategię na żywo z zapisanych prognoz.
5. **O projekcie:** cel, metoda, schemat walk-forward, ograniczenia, zastrzeżenie: "To nie jest porada inwestycyjna".
Aplikacja ma działać na Streamlit Community Cloud z `requirements-app.txt`. Przy wdrożeniu sprawdź limity darmowego planu
dla prywatnych repozytoriów. Plan awaryjny: publiczne repozytorium "demo" zawierające tylko `app/`, `results/`
i `data/raw/` (bez kodu eksperymentu), albo prezentacja lokalna.

### Wygląd aplikacji
Inspiracja: minimalistyczne strony produktowe w stylu option-o.com (jasne tło, dużo przestrzeni, mało ozdobników).
Chodzi o nastrój, nie o kopiowanie. Środki mają być proste, żeby zespół rozumiał, jak to powstało:
- Tylko wbudowane komponenty Streamlit i jeden plik `.streamlit/config.toml` z motywem (jasne tło, ciemnoszary tekst,
  jeden kolor akcentu, np. głęboki granat). Bez własnego HTML/CSS poza co najwyżej kilkoma linijkami w jednym miejscu.
- Czcionka bezszeryfowa z motywu Streamlit; jeden duży tytuł, krótkie podtytuły wersalikami lub małą czcionką, reszta tekstu zwykła.
- Wykresy Plotly z szablonem `plotly_white`, bez siatki pionowej, jedna seria w kolorze akcentu, pozostałe w szarościach,
  dni nietypowe jako dyskretne punkty.
- Układ: szeroka kolumna, duże marginesy, na górze każdej zakładki jedno zdanie wyjaśniające, co widać.
- Liczby kluczowe (trafność, zwrot, obsunięcie) jako `st.metric` w jednym rzędzie.
- Bez emoji, bez animacji, bez kolorowych tabel.

### Dokumentacja
- `README.md`: cel, instalacja krok po kroku na macOS, pobranie danych, uruchomienie eksperymentu,
  uruchomienie aplikacji, struktura repozytorium, jak odtworzyć wyniki.
- `docs/ARCHITEKTURA.md`: opis modułów i przepływu danych z diagramem Mermaid.
- `docs/DZIENNIK.md`: każda istotna decyzja techniczna (data, decyzja, powód).

### Ograniczenia do opisania w pracy
Jedna spółka; gotówka bez odsetek; stała prowizja bez poślizgu cenowego; transakcja po cenie zamknięcia; dane z jednego
dostawcy (kontrola ze Stooq); brak danych fundamentalnych i wiadomości; wyniki zależne od okresu 2020–2025.

## Etapy pracy

Po każdym etapie: testy, commit, krótkie podsumowanie (co zrobione, co działa, co zostało).

0. Szkielet: struktura katalogów, `.gitignore`, `requirements*.txt`, `src/config.py`, szkic README.
1. Dane: pobranie, zapis, walidacja, kontrola Stooq (jeśli są pliki), testy.
2. Cechy i cel prognozy, testy wycieku.
3. Model naiwny, ARIMA, silnik walk-forward, miary; przebieg `--quick`.
4. LSTM i strojenie. Najpierw przebieg `--quick` i pomiar czasu jednego treningu.
   Przed uruchomieniem trybów `--demo` i pełnego: oszacowanie ich czasu na podstawie pomiaru.
5. Isolation Forest, strategie, pliki wyników; uruchomienie w trybie `--demo` (wyniki na prezentację).
6. Aplikacja Streamlit na wynikach `--demo`.
5b. Pełny eksperyment (5 powtórzeń) uruchamiany osobno; nadpisuje `results/`.
7. Dokumentacja i przegląd końcowy: sprawdzenie każdego punktu tego pliku (co zrobione, co nie).
8. Zapis wytrenowanych modeli (`models/`), test odtworzenia prognoz z zapisanego modelu, kontrola
   powtarzalności wyników po ponownym pełnym przebiegu.
9. Komentarze w kodzie (co i dlaczego) oraz dokumentacja techniczna; tekst pracy pisemnej powstaje poza repozytorium.

## Źródła zweryfikowane

Istnienie każdej pozycji sprawdzono w internecie. "Do przeczytania" oznacza, że zespół jeszcze jej nie przeczytał:
możesz wskazać, do której części pracy pasuje, ale nie przypisuj jej konkretnych twierdzeń ani stron.

| Pozycja | Dostęp | Zastosowanie |
| --- | --- | --- |
| Fischer T., Krauss C. (2018), Deep learning with long short-term memory networks for financial market predictions, European Journal of Operational Research 270(2), 654–669, doi:10.1016/j.ejor.2017.11.054 | wersja robocza: iwf.rw.fau.de/files/2015/12/11-2017.pdf | wzór metodologii: LSTM, kierunek ruchu cen, test poza próbą |
| Hochreiter S., Schmidhuber J. (1997), Long Short-Term Memory, Neural Computation 9(8), 1735–1780, doi:10.1162/neco.1997.9.8.1735 | biblioteka | definicja LSTM |
| Fama E. (1970), Efficient Capital Markets: A Review of Theory and Empirical Work, Journal of Finance 25(2), 383–417 | biblioteka | hipoteza rynku efektywnego, interpretacja wyniku negatywnego |
| Hyndman R.J., Athanasopoulos G., Forecasting: Principles and Practice | darmowa: otexts.com | model naiwny, ARIMA, ocena prognoz, walidacja szeregów czasowych |
| Liu F.T., Ting K.M., Zhou Z.-H. (2008), Isolation Forest, Eighth IEEE International Conference on Data Mining (ICDM 2008) | biblioteka | Isolation Forest |
| Goodfellow I., Bengio Y., Courville A., Deep Learning. Systemy uczące się, PWN, Warszawa 2018, ISBN 978-83-01-19583-0 | biblioteka | teoria sieci rekurencyjnych, regularyzacja, wczesne zatrzymanie |
| Chollet F., Deep Learning. Praca z językiem Python i biblioteką Keras, Helion 2019 | biblioteka / zakup | Keras, prognozowanie szeregów czasowych |
| Géron A., Uczenie maszynowe z użyciem Scikit-Learn, Keras i TensorFlow, wyd. III, Helion | biblioteka / zakup | praktyczna budowa i uczenie modeli |
| Tadeusiewicz R., Szaleniec M., Leksykon sieci neuronowych, Wrocław 2015, licencja CC BY-SA 3.0 PL | darmowy: jbc.bj.uj.edu.pl/Content/340901 | polskie definicje pojęć |
| Tadeusiewicz R., Gąciarz T., Borowik B., Leper B., Odkrywanie właściwości sieci neuronowych przy użyciu programów w języku C#, PAU, Kraków 2007 | darmowy: bibliotekanauki.pl/books/1377771.pdf | podstawy sieci neuronowych |
| Kozdraj T., Zastosowanie sztucznych sieci neuronowych do prognozowania cen papierów wartościowych | darmowy: dspace.uni.lodz.pl/handle/11089/7700 | polskie tło: sieci neuronowe a ceny akcji |
| Morkowski J., rozprawa doktorska o prognozowaniu kierunku zmian kursu walutowego sieciami MLP, ELM i LSTM | darmowa: wbc.poznan.pl | prognozowanie naiwne, LSTM, prognoza kierunku |
| Malczewski J. (2023), Modele i algorytmy głębokiego uczenia w analizie szeregów czasowych cen mleka w proszku, rozprawa doktorska, Federacja WSB-DSW Merito | darmowa: bip.federacja.merito.pl/attachments/115/download | polskie omówienie ARIMA, LSTM i miar błędu |
| Bergmeir C., Benítez J.M. (2012), On the use of cross-validation for time series predictor evaluation, Information Sciences 191, 192–213 | biblioteka | uzasadnienie walk-forward zamiast k-fold |
| Dietterich T.G. (1998), Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms, Neural Computation 10(7), 1895–1923 | darmowy PDF: web.engr.oregonstate.edu/~tgd | uzasadnienie testu McNemara |
| Sommerville I., Inżynieria oprogramowania, wyd. X, PWN, ISBN 978-83-01-21259-9 | biblioteka | architektura, testowanie, wymagania |
| Madeyski L., Ochodek M. (red.), Inżynieria oprogramowania, 2014 | darmowy PDF: madeyski.e-informatyka.pl | testowanie i jakość kodu, po polsku |
| Dokumentacja pytest, Keras, scikit-learn, statsmodels (wersje z `run_info.json`, data dostępu) | darmowa online | opis narzędzi |
| Świercz W., Szostak R. (2024), Comparative Analysis of Predictive Models in Stock Market Price Forecasting, Debiuty Studenckie, UE Wrocław, doi:10.15611/2024.90.1.05 | darmowy: dbc.wroc.pl | przykład studencki; pułapka prognozowania poziomu ceny |
