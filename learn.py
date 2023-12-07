# Prognozowanie szeregów czasowych przy użyciu rekurencyjnych sieci neuronowych (RNN) w TensorFlow
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.src.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, GRU
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay

# ustawienie skalera do skalowania w zakresie 0-1
scaler = MinMaxScaler(feature_range=(0, 1))


# Funkcja do stworzenia zestawu danych
def create_dataset(data):
    x, y = [], []
    # petla przechodzi przez dane treningowe zaczynajac od 50 indeksu (zakladajac ze chcemy uzywac 50 poprzednich
    # punktow czasowych do prognozowania)
    for i in range(50, len(data)):
        x.append(data[i - 50:i, 0])  # dodaje 50 poprzednich punktow jako dane wejsciowe (cechy)
        y.append(data[i, 0])  # dodaje nastepne punkty czasowe jako wartosc wyjsciowa (co chcemy prognozowac)
        # Wyswietl zestawy danych
        if i <= 51:
            print("Zestawy danych \n")
            print(x)
            print(y)
            print()
    return x, y


# Funkcja do przygotowania danych do nauki
# data - dane szeregow czasowych
# percent - procentowy podzial danych na zbior treningowy i testowy
def set_data_for_learn(data, train_percent, val_percent):
    # Zapisz calkowita ilosc danych
    total_length = len(data)
    # Oblicz wielkosc danych uczacych (ceil zaokragla wynik w gore do calkowitej)
    training_data_lenght = math.ceil(total_length * train_percent)
    # Oblicz wielkosc danych walidacyjnych
    val_data_length = math.ceil(total_length * val_percent)

    # Podzial danych na trenujace, testujace i walidacyjne
    # iloc wybiera kolumne danych po to by upewnic się że mamy tylko dane wejsciowe
    train_data = data[:training_data_lenght].iloc[:, :1]  # zawiera pierwsze wiersze do ilosci okrslonej przez training_data_lenght
    val_data = data[training_data_lenght:training_data_lenght + val_data_length].iloc[:, :1] # zawiera dane od train_data i konczy na train data + val
    test_data = data[training_data_lenght + val_data_length:].iloc[:, :1]  # zawiera wszystkie pozostale (te dane, ktore nie sa uzywane do trenowania i walidacji. Model testuje sie na nieznanych danych w celu oceny jego skutecznosci
    # Wyswietl podzial danych
    print(f"Dane trenujace {train_data.shape} Dane walidacyjne {val_data.shape} Dane testowe {test_data.shape}")

    # Utworz nowa tablice i wypelnij ja wartosciami kolumny Open
    select_data_train = train_data.Open.values
    select_data_val = val_data.Open.values
    select_data_test = test_data.Open.values
    # Przeksztalcenie tablicy na dwa wymiary (bo taka moze byc uzywana jako wejscie do modelu RNN)
    # -1 oznacza ze liczba wierszy jest automatycznie dostosowana a 1 ilosc ilosc kolumn
    select_data_train = np.reshape(select_data_train, (-1, 1))
    select_data_val = np.reshape(select_data_val,(-1, 1))
    select_data_test = np.reshape(select_data_test, (-1, 1))
    # Wyswietl przeksztalcone dane
    print("Przeksztalcone dane \n")
    print(select_data_train.shape)
    print(select_data_val.shape)
    print(select_data_test.shape)
    # Normalizacja danych aby zwiekszyc efektywnosc
    # uzycie minmaxscaler w celu skalowania zbioru danych od 0 do 1.

    # Skalowanie danych
    scaled_data_train = scaler.fit_transform(select_data_train)
    scaled_data_val = scaler.fit_transform(select_data_val)
    scaled_data_test = scaler.fit_transform(select_data_test)
    # Wyswietl 5 pierwszych rekordow
    print("Przeskalowane dane \n")
    print(*scaled_data_train[:5])
    print(*scaled_data_val[:5])
    print(*scaled_data_test[:5])

    # Podzial danych na x i y ze zbioru uczacego oraz testowego.
    # Model RNN jest rekurencyjny wiec korzysta z poprzednich punktow czasowych do prognozowania kolejnych
    # Przygotowujac zestawy danych model bedzie uczony w sekwencjach poprzednich punktow czasowych (np 50 ostatnich)
    # i ich odpowiadajacych przyszlych wartosci

    # Zestaw treningowy
    x_data_train, y_data_train = create_dataset(scaled_data_train)  # x dane wejsciowe, y dane wyjsciowe
    # Zestaw walidacyjny
    x_data_val, y_data_val = create_dataset(scaled_data_val)
    # Zestaw testowy
    x_data_test, y_data_test = create_dataset(scaled_data_test)

    # Konwersja do formatu dla RNN
    x_data_train, y_data_train= np.array(x_data_train), np.array(y_data_train)
    x_data_val, y_data_val = np.array(x_data_val), np.array(y_data_val)
    x_data_test, y_data_test = np.array(x_data_test), np.array(y_data_test)

    # Przeksztalcenie wymiarow
    # X - dane wejsciowe, 3 wymiary, 1 - liczba probek, 2 - liczba krokow czasowych (w tym
    # przypadku 50, poniewaz uzywamy 50 poprzednich punktow do prognozowania), 3 - liczba cech
    x_data_train = np.reshape(x_data_train, (x_data_train.shape[0], x_data_train.shape[1], 1))
    # Y - wartosci wyjsciowe dla kazdej sekwencji x, 2 wymiary, 1 - liczba probek, 2 -  liczba krokow czasowych w
    # sekwencji = 1 poniewaz przewidujemy tylko jeden punkt czasowy do przodu
    y_data_train = np.reshape(y_data_train, (y_data_train.shape[0], 1))

    x_data_val = np.reshape(x_data_val, (x_data_val.shape[0], x_data_val.shape[1], 1))
    y_data_val = np.reshape(y_data_val, (y_data_val.shape[0], 1))

    x_data_test = np.reshape(x_data_test, (x_data_test.shape[0], x_data_test.shape[1], 1))
    y_data_test = np.reshape(y_data_test, (y_data_test.shape[0], 1))

    print("x_data_train :", x_data_train.shape, "y_data_train :", y_data_train.shape)
    print("x_data_val :", x_data_val.shape, "y_data_val :", y_data_val.shape)
    print("x_data_test :", x_data_test.shape, "x_data_test :", y_data_test.shape)

    # Zwrocenie danych
    return train_data, val_data, test_data, x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test


# Funkcja do obslugi rekurencyjnej sieci neuronowej
# Model RNN z warstwa GRU w Keras. Zawiera 4 warstwy GRU i 1 wyjsciowa
# Funkcja aktywacji tanh. W celu nadmiernego uczenia uzywa warstwy dropout
# Uzywa opytmalizatora SGD (0.01 uczenia, 1e-7 zaniku, ped 0.9, Nesterov - False
# Funkcja straty jest blad sredniokwadratowy
# Jako metryke oceny uzyto dokladnosci
# Model trenowany na danych treningowych przez 20 epok, uzywajac partii o wielkosci 2
def rnn_GRUmodel(x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test, dropout=0.2, learning_rate=0.01, momentum=0.9, epochs=20, batch_size=2, decay_steps=10000, decay_rate=0.9 ):
    # Inicjalizacja RNN
    rnn = Sequential()

    # Dodanie warstw GRU i warstwy rezygnacji,
    # funkcja aktywacja to tanh czyli tangent hiperboliczny (przeksztalca wartosci wejsciowe na zakres -1 do 1)
    # units - liczba nueronow w warstwie rekurencyjnej (wiecej - pozwoli trudniejsze dane ale moze prowadzic do nadmiernego dopasowania)
    # retrun_sequences - okresla czy warstwa rekurencyjna powinna zwracac sekwencje (wyniki kazdego kroku) czy tylko ostatni wynik
    rnn.add(GRU(units=50, return_sequences=True, input_shape=(x_data_train.shape[1], 1), activation='tanh'))
    # warstwa rezygnacji w celu zapobiegnieciu nadmiarnemu dopasowaniu (0.2 oznacza ze okolo 20% neuronow zostanie losowo wyzerowanych podczas kazdej iteracji treningowej)
    rnn.add(Dropout(dropout))

    rnn.add(GRU(units=50, return_sequences=True, activation='tanh'))
    rnn.add(GRU(units=50, return_sequences=True, activation='tanh'))
    rnn.add(GRU(units=50, activation='tanh'))

    # Dodanie warstwy wyjsciowej
    # relu to funkcja aktywacji ktora zwraca maksimum z 0 i danej wartosci wejsciowej, eliminuje wartosci ujemne
    rnn.add(Dense(units=1, activation='relu'))

    # Kompilacja rnn

    # Ponizsza funkcja jest to harmonogram szybkosci uczenia ktory jest wykladniczy
    # initial - poczatkowa szybkosc uczenia (poczatkowy kork o jaki model aktualizuje wagi podczas uczenia)
    # decay - liczba krokow po ktorej nastepuje zmniejszenie szybkosci uczenia
    # decay_rate - wspolczynnik wykladniczy o ktory zmniejsza sie szybkosc uczenia
    learning_rate_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate)
    # algorytm optymalizacji SGD
    # learn - szybkosc uczenia
    # momentum - wspolczynnik momentum (technika optymalizacji do szybszego uczenia, ignoruje nagle zmiany gradientu)
    # nesterov - okresla czy ma byc uzywane ulepszone momentum nesterova
    optimizer = SGD(learning_rate=learning_rate_schedule, momentum=momentum, nesterov=False)
    # kompilacja modelu
    # loss - funkcja straty ustawiona jako blad sredniokwadratowy poniewaz jest on odpowiedni dla problemow regresji do prognozwania ciaglych wartosci
    rnn.compile(optimizer=optimizer, loss='mean_squared_error')

    # Dopasowanie danych
    # fit - okresla jak model ma byc trenowany
    # epochs - liczba epok (ile razy model zoaczy kazdy punkt danych treningowych raz)
    # batch - liczba probek uzywanych do jadnej aktualizacji wag
    # val - dane uzywane do walidacji modelu
    rnn.fit(x_data_train, y_data_train, epochs=epochs, batch_size=batch_size, validation_data=(x_data_val, y_data_val))

    # Zapisz model po trenowaniu
    rnn.save("model_GRU.keras")

    # Wczytaj model
    # loaded_rnn = load_model("model_GRU.keras")

    rnn.summary() # Wyswietla podsumowanie architektury modelu


    # Predykcja z wykorzystaniem danych testowych
    y_rnn = rnn.predict(x_data_test)

    # Przywracanie skali z 0-1 do orginalnej
    y_rnn_org = scaler.inverse_transform(y_rnn)

    # Zwrocenie danych poddanych predykcji w orginalnej skali
    return y_rnn_org


def rnn_LSTMmodel(x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test, epochs=12, batch_size=2):
    # Inicjalizacja modelu
    rnn = Sequential()
    # Dodawanie warstw LSTM
    rnn.add(LSTM(50, return_sequences=True, input_shape=(x_data_train.shape[1], 1)))
    rnn.add(LSTM(50,return_sequences=False))
    rnn.add(Dense(25))

    # Dodanie warstwy wyjsciowej
    rnn.add(Dense(1))

    # Kompilacja modelu
    rnn.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])

    # Dopasowanie modelu
    rnn.fit(x_data_train, y_data_train, batch_size=batch_size, epochs=epochs, validation_data=(x_data_val, y_data_val))

    # Zapisz model po trenowaniu
    rnn.save("model_LSTM.keras")

    rnn.summary()  # Wyswietla podsumowanie architektury modelu

    # Predykcja z wykorzystaniem danych testowych
    y_rnn = rnn.predict(x_data_test)

    # Przywracanie skali z 0-1 do orginalnej
    y_rnn_org = scaler.inverse_transform(y_rnn)

    # Zwrocenie danych poddanych predykcji w orginalnej skali
    return y_rnn_org

# Funkcja do przedstawienia predykcji za pomoca wykresu
def draw_predict(train_data, val_data, test_data, rnn_model):

    # Wykres
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index[150:], train_data.Open[150:], label="train_data", color="b")
    plt.plot(val_data.index, val_data.Open, label="val_data", color="g")
    plt.plot(test_data.index, test_data.Open, label="test_data", color="y")
    plt.plot(test_data.index[50:], rnn_model, label="y_GRU", color="r")

    # Dodanie legendy, tytułu i etykiet osi
    plt.legend()
    plt.title("GRU Predictions")
    plt.xlabel("Days")
    plt.ylabel("Open Price")

    # Wyświetlenie wykresu
    plt.show()
