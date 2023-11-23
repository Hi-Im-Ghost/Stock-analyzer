# Prognozowanie szeregów czasowych przy użyciu rekurencyjnych sieci neuronowych (RNN) w TensorFlow
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Funkcja do przygotowania danych do nauki
# data - dane szeregow czasowych
# percent - procentowy podzial danych na zbior treningowy i testowy
def set_data_for_learn(data, percent):
    # Oblicz wielkosc danych uczacych (ceil zaokragla wynik w gore do calkowitej)
    training_data_lenght = math.ceil(len(data) * percent)
    # Podzial danych na trenujace i testujace
    # iloc wybiera kolumne danych
    train_data = data[:training_data_lenght].iloc[:,
                 :1]  # zawiera pierwsze wiersze do ilosci okrslonej przez training_data_lenght
    test_data = data[training_data_lenght:].iloc[:, :1]  # zawiera wszystkie pozostale
    # Wyswietl podzial danych
    print(f"Dane trenujace {train_data.shape} Dane testowe {test_data.shape}")

    # Utworz nowa tablice i wypelnij ja wartosciami kolumny Open
    select_data_train = train_data.Open.values
    select_data_test = test_data.Open.values
    # Przeksztalcenie tablicy na dwa wymiary (bo taka moze byc uzywana jako wejscie do modelu RNN)
    # -1 oznacza ze liczba wierszy jest automatycznie dostosowana a 1 ilosc ilosc kolumn
    select_data_train = np.reshape(select_data_train, (-1, 1))
    select_data_test = np.reshape(select_data_test, (-1, 1))
    # Wyswietl przeksztalcone dane
    print(select_data_train.shape)
    print(select_data_test.shape)
    # Normalizacja danych aby zwiekszyc efektywnosc
    # uzycie minmaxscaler w celu skalowania zbioru danych od 0 do 1.
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Skalowanie danych
    scaled_data_train = scaler.fit_transform(select_data_train)
    scaled_data_test = scaler.fit_transform(select_data_test)
    # Wyswietl 5 pierwszych rekordow
    print(*scaled_data_train[:5])
    print(*scaled_data_test[:5])

    # Podzial danych na x i y ze zbioru uczacego oraz testowego.
    # Model RNN jest rekurencyjny wiec korzysta z poprzednich punktow czasowych do prognozowania kolejnych
    # Przygotowujac zestawy danych model bedzie uczony w sekwencjach poprzednich punktow czasowych (np 50 ostatnich)
    # i ich odpowiadajacych przyszlych wartosci

    # Zestaw treningowy
    x_data_train = []  # dane wejsciowe
    y_data_train = []  # dane wyjsciowe
    # petla przechodzi przez dane treningowe zaczynajac od 50 indeksu (zakladajac ze chcemy uzywac 50 poprzednich
    # punktow czasowych do prognozowania)
    for i in range(50, len(scaled_data_train)):
        x_data_train.append(scaled_data_train[i - 50:i, 0])  # dodaje 50 poprzednich punktow jako dane wejsciowe (cechy)
        y_data_train.append(
            scaled_data_train[i, 0])  # dodaje nastepne punkty czasowe jako wartosc wyjsciowa (co chcemy prognozowac)
        # Wyswietl pierwsze dwa zestawy danych treningowych
        if i <= 51:
            print(x_data_train)
            print(y_data_train)
            print()

    # Zestaw testowy
    x_data_test = []
    y_data_test = []
    for i in range(50, len(scaled_data_test)):
        x_data_test.append(scaled_data_test[i - 50:i, 0])
        y_data_test.append(scaled_data_test[i, 0])

    # Konwersja do formatu dla RNN
    x_data_train = np.array(x_data_train)
    y_data_train = np.array(y_data_train)
    x_data_test = np.array(x_data_test)
    y_data_test = np.array(y_data_test)

    # Przeksztalcenie wymiarow
    x_data_train = np.reshape(x_data_train, (x_data_train.shape[0], x_data_train.shape[1], 1)) # 3 wymiary, 1 - liczba probek, 2 - liczba krokow czasowych (w tym przypadku 50, poniewaz uzywamy 50 poprzednich punktow do prognozowania), 3 - liczba cech
    y_data_train = np.reshape(y_data_train, (y_data_train.shape[0], 1)) # 2 wymiary, 1 - liczba probek, 2 - liczba krokow
    x_data_test = np.reshape(x_data_test, (x_data_test.shape[0], x_data_test.shape[1], 1))
    y_data_test = np.reshape(y_data_test, (y_data_test.shape[0], 1))
    print("x_data_train :", x_data_train.shape, "y_data_train :", y_data_train.shape)
    print("x_data_train :", x_data_test.shape, "y_data_train :", y_data_test.shape)
