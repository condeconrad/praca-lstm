# Propozycje źródeł do weryfikacji przez zespół

Źródła z tej listy NIE są używane w pracy, dopóki zespół ich nie potwierdzi (przeczyta i sprawdzi dane bibliograficzne).
Każda pozycja: autor, tytuł, rok, link lub DOI, do czego potrzebna.

| Propozycja | Dostęp | Do czego | Stan |
|---|---|---|---|
| Gers F.A., Schmidhuber J., Cummins F. (2000), Learning to Forget: Continual Prediction with LSTM, Neural Computation 12(10), 2451–2471 | do sprawdzenia (DOI 10.1162/089976600300015015) | pochodzenie bramki zapominania, której nie ma w pracy z 1997 r. (sekcja C1.1.7 pracy ma znacznik [DO WERYFIKACJI]) | do potwierdzenia |
| Cont R. (2001), Empirical properties of asset returns: stylized facts and statistical issues, Quantitative Finance 1(2), 223–236 | do sprawdzenia (DOI 10.1080/713665670) | "fakty stylizowane" zwrotów: grube ogony, skupianie zmienności, brak autokorelacji (pierwotna wersja pracy przywoływała to bez przypisu) | do potwierdzenia |
| Lopez de Prado M. (2018), Advances in Financial Machine Learning, Wiley | biblioteka / zakup | walidacja w finansach, wyciek danych (pierwotna wersja pracy przywoływała bez przypisu) | do potwierdzenia |

Uwaga: dokumentacja statsmodels (`ARIMAResults.append`), scikit-learn (`IsolationForest`, `StandardScaler`), Keras (`LSTM`,
`EarlyStopping`, zapis `.keras`) i scipy (`binomtest`) jest przywołana w pracy jako "dokumentacja narzędzi" z numerem wersji
z `results/run_info.json` i datą dostępu.

Wnioski z weryfikacji źródeł zweryfikowanych: Madeyski i Ochodek (2014)
nie zawiera rozdziału o testowaniu (nadaje się tylko do definicji wymagań); Bergmeir i Benítez (2012) jest dostępne
wyłącznie na poziomie abstraktu; pełny tytuł rozprawy Morkowskiego to "Metoda neuronowo-rozmyta w prognozowaniu kierunku
zmian wybranych kursów walutowych" (UEP, 2022).
