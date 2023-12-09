# Prognozowanie szeregów czasowych przy użyciu rekurencyjnych sieci neuronowych (RNN) w TensorFlow
import math

import keras.models
import numpy as np
import matplotlib.pyplot as plt
from keras.src.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, GRU
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import History

# ustawienie skalera do skalowania w zakresie 0-1
scaler = MinMaxScaler(feature_range=(0, 1))


# Funkcja do stworzenia zestawu danych
def create_dataset(data, history_size=40):
    x, y = [], []
    # petla przechodzi przez dane treningowe zaczynajac od 40 indeksu (zakladajac ze chcemy uzywac 40 poprzednich
    # punktow czasowych do prognozowania)
    for i in range(history_size, len(data)):
        x.append(data[i - history_size:i, 0])  # dodaje 40 poprzednich punktow jako dane wejsciowe (cechy)
        y.append(data[i, 0])  # dodaje nastepne punkty czasowe jako wartosc wyjsciowa (co chcemy prognozowac)
        # Wyswietl zestawy danych
        if i <= history_size + 1:
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
    # Przygotowujac zestawy danych model bedzie uczony w sekwencjach poprzednich punktow czasowych (np 40 ostatnich)
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
    # przypadku 40, poniewaz uzywamy 40 poprzednich punktow do prognozowania), 3 - liczba cech
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
    print("x_data_test :", x_data_test.shape, "y_data_test :", y_data_test.shape)

    # Zwrocenie danych
    return train_data, val_data, test_data, x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test


# Funkcja do obslugi rekurencyjnej sieci neuronowej
# Model RNN z warstwa GRU w Keras. Zawiera 4 warstwy GRU i 1 wyjsciowa
# Funkcja aktywacji tanh. W celu nadmiernego uczenia uzywa warstwy dropout
# Uzywa opytmalizatora SGD
# Funkcja straty jest blad sredniokwadratowy
# Model trenowany na danych treningowych przez 30 epok, ilosc przykladow treningowych (batch_size) = 16
def rnn_GRUmodel(x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test, dropout=0.1, learning_rate=0.01, momentum=0.9, epochs=30, batch_size=16, decay_steps=10000, decay_rate=0.9):
    # Inicjalizacja RNN
    rnn = Sequential()

    # Dodanie warstw GRU i warstwy rezygnacji,
    # funkcja aktywacja to tanh czyli tangent hiperboliczny (przeksztalca wartosci wejsciowe na zakres -1 do 1)
    # units - liczba nueronow w warstwie rekurencyjnej (wiecej - pozwoli trudniejsze dane ale moze prowadzic do nadmiernego dopasowania)
    # retrun_sequences - okresla czy warstwa rekurencyjna powinna zwracac sekwencje (wyniki kazdego kroku) czy tylko ostatni wynik
    rnn.add(GRU(units=64, return_sequences=True, input_shape=(x_data_train.shape[1], 1), activation='tanh'))
    # warstwa rezygnacji w celu zapobiegnieciu nadmiarnemu dopasowaniu (0.2 oznacza ze okolo 20% neuronow zostanie losowo wyzerowanych podczas kazdej iteracji treningowej)
    rnn.add(Dropout(dropout))

    rnn.add(GRU(units=64, return_sequences=True, activation='tanh'))
    rnn.add(GRU(units=64, return_sequences=True, activation='tanh'))
    rnn.add(GRU(units=64, activation='tanh'))
    # warstwa gesta
    # relu to funkcja aktywacji ktora zwraca maksimum z 0 i danej wartosci wejsciowej, eliminuje wartosci ujemne
    rnn.add(Dense(units=50, activation='relu'))

    # Dodanie warstwy wyjsciowej
    rnn.add(Dense(units=1))

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
    rnn.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error', 'accuracy'])

    # Obiekt history, ktory przechowuje dane treningu
    history = History()

    # Dopasowanie danych
    # fit - okresla jak model ma byc trenowany
    # epochs - liczba epok (ile razy model zoaczy kazdy punkt danych treningowych raz)
    # batch - liczba probek uzywanych do jadnej aktualizacji wag
    # val - dane uzywane do walidacji modelu
    rnn.fit(x_data_train, y_data_train, epochs=epochs, batch_size=batch_size, validation_data=(x_data_val, y_data_val), callbacks=[history])

    # Opisz proces uczenia
    describe_model(history)

    # Zapisz model po trenowaniu
    rnn.save("model_GRU.keras")

    rnn.summary() # Wyswietla podsumowanie architektury modelu


    # Predykcja z wykorzystaniem danych testowych
    y_rnn = rnn.predict(x_data_test)

    # Przywracanie skali z 0-1 do orginalnej
    y_rnn_org = scaler.inverse_transform(y_rnn)

    # Zwrocenie danych poddanych predykcji w orginalnej skali
    return y_rnn_org

# Funkcja tworzy model LSTM
# Sklada sie z 3 warstw wejsciowych i 1 wyjsciowej typu dense.
# Uzywa optymalizatora Adam
# Funkcja starty to blad sredniokwadratowy
# Szkolenie przez 20 epok, ilosc przykladow treningowych (batch_size) = 32
def rnn_LSTMmodel(x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test, epochs=30, batch_size=32):
    # Inicjalizacja modelu
    rnn = Sequential()
    # Dodawanie warstw LSTM
    rnn.add(LSTM(128, return_sequences=True, input_shape=(x_data_train.shape[1], 1)))
    rnn.add(LSTM(128,return_sequences=False))
    rnn.add(Dense(64, activation='relu'))

    # Dodanie warstwy wyjsciowej
    rnn.add(Dense(1))
    history = History()
    # Kompilacja modelu
    rnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error', 'accuracy'])

    # Dopasowanie modelu
    rnn.fit(x_data_train, y_data_train, batch_size=batch_size, epochs=epochs, validation_data=(x_data_val, y_data_val), callbacks=[history])

    # Opisz proces uczenia
    describe_model(history)

    # Zapisz model po trenowaniu
    rnn.save("model_LSTM.keras")

    rnn.summary()  # Wyswietla podsumowanie architektury modelu

    # Predykcja z wykorzystaniem danych testowych
    y_rnn = rnn.predict(x_data_test)

    # Przywracanie skali z 0-1 do orginalnej
    y_rnn_org = scaler.inverse_transform(y_rnn)

    # Zwrocenie danych poddanych predykcji w orginalnej skali
    return y_rnn_org
# Funkcja do wizualizacji procesu uczenia
# Przedstawia wykres dokladnosci, straty, bledu uczenia
def describe_model(history):

    # Wykres dokładnosci (accuracy) w kolejnych epokach
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Wykres funkcji straty (loss) w kolejnych epokach
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Wykres bledu uczenia w kolejnych epokach
    plt.plot(history.history['mean_squared_error'], label='Train MSE')
    plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
    plt.title('Mean Squared Error Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

    # Wykres mean_absolute_error
    plt.plot(history.history['mean_absolute_error'], label='Train MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Mean Absolute Error Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()
def load_model(path, x_data_test):

    # Wczytaj model
    loaded_rnn = keras.models.load_model(path)

    loaded_rnn.summary() # Wyswietla podsumowanie architektury modelu


    # Predykcja z wykorzystaniem danych testowych
    y_rnn = loaded_rnn.predict(x_data_test)

    # Przywracanie skali z 0-1 do orginalnej
    y_rnn_org = scaler.inverse_transform(y_rnn)

    return y_rnn_org


# Funkcja do przedstawienia predykcji za pomoca wykresu
def draw_predict(type, train_data, val_data, test_data, rnn_model):

    # Wykres
    plt.figure(figsize=(14, 6))
    plt.plot(train_data.index, train_data.Open, label="train_data", color="b")
    plt.plot(val_data.index, val_data.Open, label="val_data", color="g")
    plt.plot(test_data.index, test_data.Open, label="test_data", color="y")
    plt.plot(test_data.index[40:], rnn_model, label=type, color="r") #index zalezy od history_size czyli poprzednich puntkow czasowych

    # Dodanie legendy, tytułu i etykiet osi
    plt.legend()
    plt.title("Predictions " + type)
    plt.xlabel("Days")
    plt.ylabel("Open Price")

    # Wyświetlenie wykresu
    plt.show()
