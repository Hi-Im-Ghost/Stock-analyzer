import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import math


# Funkcja do regresji liniowej
def linear_regression(data, independent_variable, dependent_variable):
    if independent_variable not in data.columns or dependent_variable not in data.columns:
        print("Niektóre z podanych kolumn nie istnieją.")
        return

    x = data[independent_variable].values
    y = data[dependent_variable].values

    # Oblicz współczynniki regresji przy użyciu wzoru regresji liniowej
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy_mean = np.mean(x * y)
    x_squared_mean = np.mean(x ** 2)

    slope = (xy_mean - x_mean * y_mean) / (x_squared_mean - x_mean ** 2)
    intercept = y_mean - slope * x_mean

    # Oblicz przewidywane wartości 'close' na podstawie 'open'
    predicted_close = slope * x + intercept

    print(f"Równanie regresji liniowej: close = {slope:.4f} * open + {intercept:.4f}")

    # Wygeneruj wykres
    plt.scatter(x, y, s=2, label='Pomiary')
    plt.plot(x, predicted_close, color='red', label='Regresja liniowa')
    plt.xlabel(independent_variable)
    plt.ylabel(dependent_variable)
    plt.legend()
    plt.show()


# Funkcja do wyznaczenia wspolczynnika korelacji liniowej Pearsona
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

        column_data = data[column]

        # Zakres min-max
        min_value = column_data.min()
        max_value = column_data.max()

        # Wartosc srednia i odchylenie
        mean = column_data.mean()
        std_dev = column_data.std()

        # Mediana
        median = column_data.median()

        # Rostep miedzykwartylowy
        q1 = column_data.quantile(0.25)
        q3 = column_data.quantile(0.75)
        iqr = q3 - q1

        # Kwantyle rzędu 0.1 i 0.9
        quantile_01 = column_data.quantile(0.1)
        quantile_09 = column_data.quantile(0.9)

        # Wartości graniczne dla punktów oddalonych
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identyfikacja punktów oddalonych
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
    correlation_matrix = calculate_pearson_correlation(data, use_cols)
    if correlation_matrix is not None:
        print("Macierz korelacji Pearsona:")
        print(correlation_matrix)

    # Wyznacz regresje liniowa
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
    data[use_cols[1:]].plot(figsize=(10, 6))
    plt.title("Wykres Wartości Zmiennych")
    plt.xlabel("Rok")
    plt.ylabel("Wartość")
    plt.show()

    # Ogólny histogram zmiennych
    data[use_cols[1:]].plot.hist(bins=10, figsize=(10, 6))
    plt.title("Histogram Zmiennych")
    plt.xlabel("Wartość")
    plt.ylabel("Częstotliwość")
    plt.show()

    # Wykres pudełkowy
    data['Year'] = pd.to_datetime(data.index).year  # Tworzenie kolumny 'Year' na podstawie roku z daty
    boxplot_cols = ['Year'] + use_cols[1:]
    data[boxplot_cols].boxplot(by='Year', figsize=(12, 6))
    plt.suptitle("Wykres Pudełkowy")
    plt.show()

    # Opisz dane
    describe_data(data, use_cols)
    # Zapisz przetworzone dane do nowego pliku
    data.to_csv(output_data)


usecols_btc = ["Date", "Open", "High", "Low", "Close"]
check_data('bitcoin.csv', 'new_bitcoin.csv', usecols_btc)
