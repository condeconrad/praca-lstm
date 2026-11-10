# Pomysły na rozszerzenia (poza zakresem pracy)

- QQQ (NASDAQ-100) jako drugi obiekt badania: czy wyniki z NVDA powtarzają się na szerokim rynku. Kod obsługuje listę tickerów.
- Druga spółka o innym charakterze (np. Apple) dla porównania zmienności.
- Test na żywo: codzienna prognoza zapisywana od lutego do obrony.
- Test Diebolda-Mariano do porównania błędów prognoz.
- Ponowny trening LSTM na całym oknie uczącym (bez 20% walidacji) ze stałą liczbą epok ustaloną w pierwszym treningu.
- Zapis krzywych uczenia (strata na części uczącej i walidacyjnej) dla każdego okna, do wykresu w pracy.
- Porównanie rzędów ARIMA: wymuszony AR(1) obok wyboru po AIC (który daje (0,0,0)).
