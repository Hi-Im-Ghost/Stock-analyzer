import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Funkcja do regresji liniowej - metoda najmniejszych kwadratow (minimalizacja sumy kwadratow, roznic pomiedzy wartosciami)
# przyjmuje dane oraz nazwy kolumn dla zmiennej niezaleznej i zaleznej
def linear_regression(data, independent_variable, dependent_variable):
    # sprawdzenie czy podane kolumny istnieja w danych
    if independent_variable not in data.columns or dependent_variable not in data.columns:
        print("Niektóre z podanych kolumn nie istnieją.")
        return

    # inicjalizacja zmiennych
    x = data[independent_variable].values  # Open
    y = data[dependent_variable].values  # Close

    # Wyliczanie potrzebnych statystyk
    # Srednie
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy_mean = np.mean(x * y)
    x_squared_mean = np.mean(x ** 2)  # srednia wartosc kwadratu

    # Wyznaczenie wspolczynnikow regresji
    # Wspolczynnik nachylenia linii regresji - okresla jak bardzo zmienia sie zmienna zalezna wraz ze zmiana niezaleznej
    slope = (xy_mean - x_mean * y_mean) / (x_squared_mean - x_mean ** 2)
    # Punkt przeciecia linii regresji z osia y czyli wartosc y gdy x wynosi 0
    intercept = y_mean - slope * x_mean

    # Oblicz przewidywane wartości 'close' na podstawie 'open'
    predicted_close = slope * x + intercept

    print(f"Równanie regresji liniowej: close = {slope:.4f} * open + {intercept:.4f}")

    # Wygeneruj wykres punktowy
    plt.scatter(x, y, s=2, label='Pomiary')  # x - niezalezna y - zalezna s - rozmiar punktow na wykresie
    plt.plot(x, predicted_close, color='red', label='Regresja liniowa') # utworzenie linii regresji na podstawie uzyskanych wspolczynnikow
    plt.xlabel(independent_variable)
    plt.ylabel(dependent_variable)
    plt.legend()
    plt.show()


# Funkcja do wyznaczenia wspolczynnika korelacji liniowej Pearsona - funkcja uzywa metody .corr() z biblioteki pandas
# przyjmuje dane z data i liste kolum dla ktorych chcemy obliczyc korelacje
# zwraca macierz korelacji gdzie wartosc pozycji i,j to korelacja miedzy kolumnami
def calculate_pearson_correlation(data, use_cols):
    correlation_matrix = data[use_cols[1:]].corr(method='pearson')
    return correlation_matrix


# Funkcja do opisu danych
def describe_data(data, use_cols):
    # Sprawdzenie czy dane maja te kolumny
    for column in use_cols[1:]:
        if column not in data.columns:
            print(f"Kolumna {column} nie istnieje w pliku CSV.")
            continue

        # Zapisanie danych z wybranej kolumny
        column_data = data[column]

        # Zakres min-max
        min_value = column_data.min()
        max_value = column_data.max()

        # Wartosc srednia i odchylenie
        mean = column_data.mean()
        std_dev = column_data.std()

        # Mediana
        median = column_data.median()

        # Rostep miedzykwartylowy - okresla zmiennosc danych w ich srodkowej czesci
        # Pomaga mierzyc jak szeroki jest zakres w ktorym skupiona jest wiekszosc danych ignorujac skrajne
        q1 = column_data.quantile(
            0.25)  # pierwszy kwartyl - 25 percentyl (wartosc ponizej ktorej znajduje sie 25% danych czyli 25% obserwacji)
        q3 = column_data.quantile(
            0.75)  # drugi kwartyl  to 75 percentyli (wskazuje gdzie zaczyna sie 3 czesc danych zawierajaca 25% obserwacji)
        iqr = q3 - q1  # rozstep

        # Kwantyle rzędu 0.1 i 0.9 - umozliwia zrozumienie w jakim stopniu dane sa skupione w roznych obszarach
        quantile_01 = column_data.quantile(0.1)  # wyznacza wartosci ponizej ktorych zawiera sie 10% danych
        quantile_09 = column_data.quantile(0.9)  # wyznacza wartosci ponizej ktorych zawiera sie 90% danych

        # Wartości graniczne dla punktów oddalonych - sluzy w identyfikacji potencjalnych bledow w danych lub obserwacji
        lower_bound = q1 - 1.5 * iqr  # polozenie ponizej
        upper_bound = q3 + 1.5 * iqr  # polozenie powyzej

        # Identyfikacja punktów oddalonych - punkty ktore znajduja sie poza granicami wyznaczonymi na podstawie iqr
        outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]

        print(f"Statystyki dla kolumny '{column}':")
        print(f"Zakres wartości (min-max): {min_value} - {max_value}")
        print(f"Wartość średnia: {mean}")
        print(f"Odchylenie standardowe: {std_dev}")
        print(f"Mediana: {median}")
        print(f"Rozstęp międzykwartylowy (IQR): {iqr}")
        print(f"Kwantyl 0.1: {quantile_01}")
        print(f"Kwantyl 0.9: {quantile_09}")
        print("\n")

        print(f"Punkty oddalone w kolumnie '{column}':")
        if len(outliers) > 0:
            print(outliers)
        else:
            print("Brak punktów oddalonych w tej kolumnie.")
        print("\n")

    # Oblicz macierz korelacji Pearsona
    # Wartosc korelacji okresla sile zaleznosc im blizej 1 tym silniejsza zaleznosc
    # Pomaga zrozumiec jak zmienne wplywaja na siebie nawzajem
    correlation_matrix = calculate_pearson_correlation(data, use_cols)
    if correlation_matrix is not None:
        print("Macierz korelacji Pearsona:")
        print(correlation_matrix)

    # Wyznacz regresje liniowa - Pomaga zrozumiec jak zmiana jednej zmiennej wplywa na druga oraz umozliwia
    # prognozowanie wartosci zmiennej zaleznej na podstawie obs. danych np. Prognozowanie zmiennej zaleznej Close na podstawie niezaleznej Open
    linear_regression(data, 'Open', 'Close')


# Funkcja do sprawdzenia danych
def check_data(input_data, output_data, use_cols):
    # Wczytywanie danych
    data = pd.read_csv(input_data, usecols=use_cols)
    # Wyswietl 5 poczatkowych i koncowych rekordow
    data.head()
    print(data)

    # Ustaw index
    if use_cols[0] in data.columns:
        data.set_index(use_cols[0], inplace=True)
    else:
        print(f"Kolumna {use_cols[0]} nie istnieje w pliku CSV.")
        return

    # Sprawdz reszte kolumn
    for column in use_cols[1:]:
        if column not in data.columns:
            print(f"Kolumna {column} nie istnieje w pliku CSV.")
            continue

        # Sprawdzenie danych podanej kolumny
        missing_data = data[column].isnull().values.any()
        # Jesli sa jakies braki w danych to...
        if missing_data.any():
            # Obliczenie sredniej wartosci kolumny
            mean_value = data[column].mean()
            # Wypelnij dane
            data.loc[missing_data, column] = mean_value

    # Ogólny wykres wartości zmiennych
    data[use_cols[1:]].plot(figsize=(10, 6)) # figsize rozmiar wykresu w calach (szerokosc,wysokosc)
    plt.title("Wykres Wartości Zmiennych")
    plt.xlabel("Rok")
    plt.ylabel("Wartość")
    plt.show()

    # Ogólny histogram zmiennych
    data[use_cols[1:]].plot.hist(bins=400, figsize=(10, 6)) # bins - okresla liczbe przedzialow (metoda scotta wychodzi ponad 2000 dla tych danych)
    plt.title("Histogram Zmiennych")
    plt.xlabel("Wartość")
    plt.ylabel("Częstotliwość")
    plt.show()

    # Wykres pudełkowy
    data['Year'] = pd.to_datetime(data.index).year  # Tworzenie kolumny 'Year' na podstawie roku z daty
    boxplot_cols = ['Year'] + use_cols[1:] # tworzenie kolumn uzywanych do wygenerowania wykresu
    data[boxplot_cols].boxplot(by='Year', figsize=(12, 6)) # by - grupowanie danych (co roku oznacza ze dla kazdego roku bedziemy mieli osobne pudelko)
    plt.suptitle("Wykres Pudełkowy")
    plt.show()

    # Opisz dane
    describe_data(data, use_cols)
    # Zapisz przetworzone dane do nowego pliku
    data.to_csv(output_data)
