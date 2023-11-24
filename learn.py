# Prognozowanie szeregów czasowych przy użyciu rekurencyjnych sieci neuronowych (RNN) w TensorFlow
import math
import numpy as np
from keras.src.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU, Bidirectional
from keras.optimizers import SGD
import matplotlib.pyplot as plt

scaler = MinMaxScaler(feature_range=(0, 1))

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
    x_data_train = np.reshape(x_data_train, (x_data_train.shape[0], x_data_train.shape[1],
                                             1))  # 3 wymiary, 1 - liczba probek, 2 - liczba krokow czasowych (w tym przypadku 50, poniewaz uzywamy 50 poprzednich punktow do prognozowania), 3 - liczba cech
    y_data_train = np.reshape(y_data_train,
                              (y_data_train.shape[0], 1))  # 2 wymiary, 1 - liczba probek, 2 - liczba krokow
    x_data_test = np.reshape(x_data_test, (x_data_test.shape[0], x_data_test.shape[1], 1))
    y_data_test = np.reshape(y_data_test, (y_data_test.shape[0], 1))
    print("x_data_train :", x_data_train.shape, "y_data_train :", y_data_train.shape)
    print("x_data_train :", x_data_test.shape, "y_data_train :", y_data_test.shape)

    # Zwrocenie danych
    return train_data, test_data,x_data_train, y_data_train, x_data_test, y_data_test


# Funkcja do obslugi rekurencyjnej sieci neuronowej
# Model RNN z warstwa GRU w Keras. Zawiera 4 warstwy GRU i 1 wyjsciowa
# Funkcja aktywacji tanh. W celu nadmiernego uczenia uzywa warstwy dropout
# Uzywa opytmalizatora SGD (0.01 uczenia, 1e-7 zaniku, ped 0.9, Nesterov - False
# Funkcja straty jest blad sredniokwadratowy
# Jako metryke oceny uzyto dokladnosci
# Model trenowany na danych treningowych przez 20 epok, uzywajac partii o wielkosci 2
def rnn_model(x_data_train, y_data_train, x_data_test, y_data_test):
    # Inicjalizacja RNN
    rnn = Sequential()

    # Dodanie warstw GRU i warstwy rezygnacji
    rnn.add(GRU(units=50, return_sequences=True, input_shape=(x_data_train.shape[1], 1), activation='tanh'))
    rnn.add(Dropout(0.2))

    rnn.add(GRU(units=50, return_sequences=True, activation='tanh'))

    rnn.add(GRU(units=50, return_sequences=True, activation='tanh'))

    rnn.add(GRU(units=50, activation='tanh'))

    # Dodanie warstwy wyjsciowej
    rnn.add(Dense(units=1, activation='relu'))
    # Kompilacja rnn
    learning_rate_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)
    optimizer = SGD(learning_rate=learning_rate_schedule, momentum=0.9, nesterov=False)
    rnn.compile(optimizer=optimizer, loss='mean_squared_error')

    # Dopasowanie danych
    rnn.fit(x_data_train, y_data_train, epochs=20, batch_size=1)
    rnn.summary()

    # Predykcja z wykorzystaniem danych testowych
    y_rnn = rnn.predict(x_data_test)

    # Przywracanie skali z 0-1 do orginalnej
    y_rnn_org = scaler.inverse_transform(y_rnn)

    # Zwrocenie danych poddanych predykcji w orginalnej skali
    return y_rnn_org


# Funkcja do przedstawienia predykcji za pomoca wykresu
def draw_predict(train_data, test_data, rnn_model):
    # Wykres
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index[150:], train_data.Open[150:], label="train_data", color="b")
    plt.plot(test_data.index, test_data.Open, label="test_data", color="g")
    plt.plot(test_data.index[50:], rnn_model, label="y_GRU", color="red")

    # Dodanie legendy, tytułu i etykiet osi
    plt.legend()
    plt.title("GRU Predictions")
    plt.xlabel("Days")
    plt.ylabel("Open Price")

    # Wyświetlenie wykresu
    plt.show()
