import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sigs
from numpy.fft import fft, fftshift


# EXERCÍCIO 1
# Abre o ficheiro com dados de um sinal
def open_users_data(exp, user):
    aux = []
    # Nome do ficheiro a ser aberto
    filename = "acc_exp%d_user%d.txt" % (exp, user)
    with open(filename, "r") as file:
        # Guarda a data do user no array arr
        for i in file:
            aux.append([float(i.split()[0]), float(
                i.split()[1]), float(i.split()[2])])
        arr = np.array(aux)
    return arr


def open_labels(exp, user):
    aux = []
    with open("labels.txt", "r") as labelsfile:
        for i in labelsfile:
            if int(i.split()[0]) == exp and int(i.split()[1]) == user:
                aux.append([int(i.split()[2]), int(
                    i.split()[3]), int(i.split()[4])])
        labels_arr = np.array(aux)
    return labels_arr


# EXERCÍCIO 2
# Cria os gráficos de cada sinal com as devidas atividades identificadas
def plot_signals_activities(exp, user):
    labels = open_labels(exp, user)

    # Inicializa os plots do x
    plt.figure(figsize=(20, 12))
    plt.subplot(311)
    plt.ylabel(r'ACC_X')
    plt.title('Graphic of $acc\_exp%d\_user%d.txt$' % (exp, user))
    plt.plot(tempo, x, 'black')
    # Dá plot do x
    for i in range(len(labels)):
        plt.plot(tempo[labels[i][1]:labels[i][2]],
                 x[labels[i][1]:labels[i][2]], colors[labels[i][0] - 1])
        if i % 2 == 0:
            plt.text((labels[i][1] + labels[i][2]) / (2 * frequency * 60), min(x),
                     atividades[labels[i][0] - 1], fontsize=8)
        else:
            plt.text((labels[i][1] + labels[i][2]) / (2 * frequency * 60), max(x),
                     atividades[labels[i][0] - 1], fontsize=8)

    # Inicializa os plots do y
    plt.subplot(312)
    plt.ylabel(r'ACC_Y')
    plt.plot(tempo, y, 'black')
    # Dá plot do y
    for i in range(len(labels)):
        plt.plot(tempo[labels[i][1]:labels[i][2]],
                 y[labels[i][1]:labels[i][2]], colors[labels[i][0] - 1])
        if i % 2 == 0:
            plt.text((labels[i][1] + labels[i][2]) / (2 * 50 * 60), min(y),
                     atividades[labels[i][0] - 1], fontsize=8)
        else:
            plt.text((labels[i][1] + labels[i][2]) / (2 * 50 * 60), max(y),
                     atividades[labels[i][0] - 1], fontsize=8)

    # Inicializa os plots do z
    plt.subplot(313)
    plt.xlabel(r'Time [min]')
    plt.ylabel(r'ACC_Z')
    plt.plot(tempo, z, 'black')
    # Dá plot do z
    for i in range(len(labels)):
        plt.plot(tempo[labels[i][1]:labels[i][2]],
                 z[labels[i][1]:labels[i][2]], colors[labels[i][0] - 1])
        if i % 2 == 0:
            plt.text((labels[i][1] + labels[i][2]) / (2 * 50 * 60), min(z),
                     atividades[labels[i][0] - 1], fontsize=8)
        else:
            plt.text((labels[i][1] + labels[i][2]) / (2 * 50 * 60), max(z),
                     atividades[labels[i][0] - 1], fontsize=8)


# De acordo com o utilizador pretendido é feita a separação das atividades
# realizadas e os valores são guardados nas variaveis definidas no main
def activities_separation(exp, user, n):
    labels = open_labels(exp, user)

    for i in labels:
        # Dinâmicos
        if i[0] == 1:
            w_x[n].append(x[i[1]:i[2]])
            w_y[n].append(y[i[1]:i[2]])
            w_z[n].append(z[i[1]:i[2]])
        elif i[0] == 2:
            w_u_x[n].append(x[i[1]:i[2]])
            w_u_y[n].append(y[i[1]:i[2]])
            w_u_z[n].append(z[i[1]:i[2]])
        elif i[0] == 3:
            w_d_x[n].append(x[i[1]:i[2]])
            w_d_y[n].append(y[i[1]:i[2]])
            w_d_z[n].append(z[i[1]:i[2]])

        # Estáticos
        elif i[0] == 4:
            sit_x[n].append(x[i[1]:i[2]])
            sit_y[n].append(y[i[1]:i[2]])
            sit_z[n].append(z[i[1]:i[2]])
        elif i[0] == 5:
            stand_x[n].append(x[i[1]:i[2]])
            stand_y[n].append(y[i[1]:i[2]])
            stand_z[n].append(z[i[1]:i[2]])
        elif i[0] == 6:
            lay_x[n].append(x[i[1]:i[2]])
            lay_y[n].append(y[i[1]:i[2]])
            lay_z[n].append(z[i[1]:i[2]])

        # Transição
        elif i[0] == 7:
            stand_sit_x[n].append(x[i[1]:i[2]])
            stand_sit_y[n].append(y[i[1]:i[2]])
            stand_sit_z[n].append(z[i[1]:i[2]])
        elif i[0] == 8:
            sit_stand_x[n].append(x[i[1]:i[2]])
            sit_stand_y[n].append(y[i[1]:i[2]])
            sit_stand_z[n].append(z[i[1]:i[2]])
        elif i[0] == 9:
            sit_lay_x[n].append(x[i[1]:i[2]])
            sit_lay_y[n].append(y[i[1]:i[2]])
            sit_lay_z[n].append(z[i[1]:i[2]])
        elif i[0] == 10:
            lay_sit_x[n].append(x[i[1]:i[2]])
            lay_sit_y[n].append(y[i[1]:i[2]])
            lay_sit_z[n].append(z[i[1]:i[2]])
        elif i[0] == 11:
            stand_lay_x[n].append(x[i[1]:i[2]])
            stand_lay_y[n].append(y[i[1]:i[2]])
            stand_lay_z[n].append(z[i[1]:i[2]])
        elif i[0] == 12:
            lay_stand_x[n].append(x[i[1]:i[2]])
            lay_stand_y[n].append(y[i[1]:i[2]])
            lay_stand_z[n].append(z[i[1]:i[2]])


# EXERCÍCIO 3.1 e 3.2
# Plot do sinal de cada uma das ações
def plot_DFT_activities(x, y, z, n, exp, user, activity):
    # Array com os vários nomes das janelas escolhidas para o plot da DFT
    name_window = ['blackman', 'hamming', 'hanning']

    sensors = ['X', 'Y', 'Z']
    arr = [x, y, z]

    for s in range(len(sensors)):
        # range(1) -> plot dos gráficos apenas para a primeira ocorrência da atividade
        # range(len(arr[s][n])) -> plot dos gráficos para todas as ocorrência da atividade
        for i in range(1):
            fig = plt.figure(figsize=(20, 4))
            title = 'Sinal Original da Janela - %s (%s) - $acc\_exp%d\_user%d.txt$' % (
                activity, sensors[s], exp, user)

            plot_Original(arr[s][n], i, title)

            for name in name_window:
                fig2 = plt.figure(figsize=(20, 4))
                title = '|DFT| do sinal - %s (%s) - %s - $acc\_exp%d\_user%d.txt$' % (
                    activity, sensors[s], name, exp, user)

                plot_DFT_NoDetrend(arr[s][n], i, title, name)
                fig3 = plt.figure(figsize=(20, 4))
                title = '|DFT| do sinal sem tendencia - %s (%s) - %s - $acc\_exp%d\_user%d.txt$' % (
                    activity, sensors[s], name, exp, user)

                plot_DFT_Detrend(arr[s][n], i, title, name)


def plot_Original(arr, i, title):
    window = np.array(arr[i])
    period = 1 / frequency
    n = window.size  # Período Fundamental
    t = np.linspace(0, period * (n - 1) / 60, n)
    plt.plot(t, window)
    plt.title(title)
    plt.xlabel('t[min]')
    plt.ylabel('Amplitude')
    plt.xlim(0, period * (n - 1) / 60)
    # linha de tendência
    trend = window - sigs.detrend(window, type='constant')
    plt.plot(t, trend)


def plot_DFT_NoDetrend(arr, i, title, name):
    aux = np.array(arr[i])
    if name == 'hamming':
        window = np.hamming(aux.size)
    elif name == 'blackman':
        window = np.blackman(aux.size)
    elif name == 'hanning':
        window = np.hanning(aux.size)
    X = abs(fftshift(fft(aux)) * window)
    N = X.size  # Período Fundamental
    if N % 2 == 0:  # Definição de Frequência
        f = np.arange(-frequency / 2,
                      frequency / 2, frequency / N)
    else:
        f = np.arange(-frequency / 2 + frequency / (2 * N),
                      frequency / 2 - frequency / (2 * N) + frequency / N, frequency / N)
    plt.plot(f, X)
    plt.title(title)
    plt.xlabel('F(Hz)')
    plt.ylabel('Magnitude |X|')


def plot_DFT_Detrend(arr, i, title, name):
    aux = np.array(arr[i])
    if name == 'hamming':
        window = np.hamming(aux.size)
    elif name == 'blackman':
        window = np.blackman(aux.size)
    elif name == 'hanning':
        window = np.hanning(aux.size)
    X = abs(fftshift(fft(sigs.detrend(aux))) * window)
    N = X.size  # Período Fundamental
    if N % 2 == 0:  # Definição de Frequência
        f = np.arange(-frequency / 2,
                      frequency / 2, frequency / N)
    else:
        f = np.arange(-frequency / 2 + frequency / (2 * N),
                      frequency / 2 - frequency / (2 * N) + frequency / N, frequency / N)
    plt.plot(f, X)
    plt.title(title)
    plt.xlabel('F(Hz)')
    plt.ylabel('Magnitude |X|')


def calculate_max_mag(x, y, z):
    sensors = [x, y, z]
    for s in sensors:
        for i in range(len(s)):
            X = abs(fftshift(fft(sigs.detrend(s[i]))))
            max_magnitude = X[sigs.find_peaks(X)[0]]
            max_magnitude = max(max_magnitude)
            if s == x:
                max_mag_x.append(max_magnitude)
            elif s == y:
                max_mag_y.append(max_magnitude)
            else:
                max_mag_z.append(max_magnitude)


def calculate_min_mag(x, y, z):
    sensors = [x, y, z]
    for s in sensors:
        for i in range(len(s)):
            X = abs(fftshift(fft(sigs.detrend(s[i]))))
            min_magnitude = X[sigs.find_peaks(X)[0]]
            min_magnitude = min(min_magnitude)
            if s == x:
                min_mag_x.append(min_magnitude)
            elif s == y:
                min_mag_y.append(min_magnitude)
            else:
                min_mag_z.append(min_magnitude)


def plot_windows(act):
    aux = np.array(act[0])
    window_hamming = np.hamming(aux.size)
    window_hanning = np.hanning(aux.size)
    window_blackman = np.blackman(aux.size)
    plt.figure(figsize=(15,5))
    plt.plot(window_hamming)
    plt.plot(window_hanning)
    plt.plot(window_blackman)
    plt.title("Janelas")
    plt.legend(["Hamming", "Hanning", "Blackman"])


def steps_per_min(w, wu, wd):
    total_steps_w = []
    total_steps_wu = []
    total_steps_wd = []

    for i in range(8):
        # WALKING
        for j in range(len(w[i])):
            m = abs(fftshift(fft(sigs.detrend(w[i][j]))))
            peaks, _ = sigs.find_peaks(m)
            num_steps = peaks.size
            total_steps_w.append(num_steps / 2)

        # WALKING UPSTAIRS
        for j in range(len(wu[i])):
            m = abs(fftshift(fft(sigs.detrend(wu[i][j]))))
            peaks, _ = sigs.find_peaks(m)
            num_steps = peaks.size
            total_steps_wu.append(num_steps / 2)

        # WALKING DOWNSTAIRS
        for j in range(len(wd[i])):
            m = abs(fftshift(fft(sigs.detrend(wd[i][j]))))
            peaks, _ = sigs.find_peaks(m)
            num_steps = peaks.size
            total_steps_wd.append(num_steps / 2)

    mean_steps_w = np.mean(total_steps_w)
    deviation_steps_w = np.std(total_steps_w)
    mean_steps_wu = np.mean(total_steps_wu)
    deviation_steps_wu = np.std(total_steps_wu)
    mean_steps_wd = np.mean(total_steps_wd)
    deviation_steps_wd = np.std(total_steps_wd)

    return ['%.2f +- %.2f' % (mean_steps_w, deviation_steps_w), '%.2f +- %.2f' % (
        mean_steps_wu, deviation_steps_wu), '%.2f +- %.2f' % (mean_steps_wd, deviation_steps_wd)]


def diferenciate_activities():
    fig = plt.figure(figsize=(10, 5))
    p = fig.add_subplot(projection='3d')

    for i in range(8):
        # WALKING
        for j in range(len(w_x[i])):
            window = np.hamming(len(w_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(w_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(w_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(w_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='none',
                      marker='o', edgecolor='blue')

        # WALKING UPSTAIRS
        for j in range(len(w_u_x[i])):
            window = np.hamming(len(w_u_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(w_u_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(w_u_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(w_u_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='none',
                      marker='o', edgecolor='black')

        # WALKING DOWNSTAIRS
        for j in range(len(w_d_x[i])):
            window = np.hamming(len(w_d_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(w_d_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(w_d_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(w_d_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='none',
                      marker='o', edgecolor='red')

        # SITTING
        for j in range(len(sit_x[i])):
            window = np.hamming(len(sit_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(sit_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(sit_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(sit_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='tan',
                      marker='+')

        # STAND
        for j in range(len(stand_x[i])):
            window = np.hamming(len(stand_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(stand_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(stand_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(stand_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='olive',
                      marker='+')

        # LAYING
        for j in range(len(lay_x[i])):
            window = np.hamming(len(lay_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(lay_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(lay_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(lay_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='skyblue',
                      marker='+')

        # STAND TO SIT
        for j in range(len(stand_sit_x[i])):
            window = np.hamming(len(stand_sit_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(stand_sit_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(stand_sit_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(stand_sit_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='none',
                      marker='d', edgecolor='slategrey')

        # SIT TO STAND
        for j in range(len(sit_stand_x[i])):
            window = np.hamming(len(sit_stand_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(sit_stand_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(sit_stand_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(sit_stand_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='none',
                      marker='d', edgecolor='pink')

        # SIT TO LAY
        for j in range(len(sit_lay_x[i])):
            window = np.hamming(len(sit_lay_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(sit_lay_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(sit_lay_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(sit_lay_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='none',
                      marker='d', edgecolor='gold')

        # LAY TO SIT
        for j in range(len(lay_sit_x[i])):
            window = np.hamming(len(lay_sit_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(lay_sit_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(lay_sit_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(lay_sit_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='none',
                      marker='d', edgecolor='purple')

        # STAND TO LAY
        for j in range(len(stand_lay_x[i])):
            window = np.hamming(len(stand_lay_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(stand_lay_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(stand_lay_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(stand_lay_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='none',
                      marker='d', edgecolor='orange')

        # LAY TO STAND
        for j in range(len(lay_stand_x[i])):
            window = np.hamming(len(lay_stand_x[i][j]))
            m_x = abs(fftshift(fft(sigs.detrend(lay_stand_x[i][j]) * window)))
            peaks_x, _ = sigs.find_peaks(m_x)

            m_y = abs(fftshift(fft(sigs.detrend(lay_stand_y[i][j]) * window)))
            peaks_y, _ = sigs.find_peaks(m_y)

            m_z = abs(fftshift(fft(sigs.detrend(lay_stand_z[i][j]) * window)))
            peaks_z, _ = sigs.find_peaks(m_z)
            p.scatter(m_x[min(peaks_x)], m_y[min(peaks_y)], m_z[min(peaks_z)], color='none',
                      marker='d', edgecolor='blue')
    


# Questão 4
def plot_STFT(array, exp, user):
    az = []
    for i in array:
        az.append(i[2])

    z = np.array(az, dtype=object)
    
    period = 1 / frequency  # Período fundamental
    n = z.size
    t = n * period / 60  # Tempo total da atividade do utilizador
    t_frame = 0.005 * t  # Janela de tamanho 0.005 vezes menor que o tempo de atividade do utilizador

    # Sobreposicao de 50% de modo a melhorar a continuidade temporal e minimizar a atenuacao da janelas nas epocas
    # afastadas do centro
    t_overlap = t_frame / 2
    n_frame = round(t_frame * frequency)


    h = np.hamming(n_frame)

    n_overlap = round(t_overlap * frequency)
    espetro = []  # Vetor para guardar o array do sinal
    f = np.linspace(-frequency / 2, frequency / 2, n_frame)  # Cálculo da frequência
    x = np.where(f >= 0)  # Tratamento do sinal apenas em valores positivos

    for j in range(0, n - n_frame, n_frame - n_overlap):
        x_frame = z[j : j + n_frame] * h
        m_x_frame = abs(fftshift(fft(x_frame)))
        espetro = np.hstack([espetro, m_x_frame[x]])

    plt.figure(figsize=(15, 7))
    plt.title("STFT - $acc\_exp%d\_user%d.txt$" % (exp, user))
    plt.specgram(espetro, Fs=frequency)
    plt.xlabel("Tempo [min]")
    plt.ylabel("Frequência [Hz]")
    plt.colorbar()


if __name__ == "__main__":
    # Variáveis necessárias para importar data de diferentes ficheiros de dados
    user = [21, 22, 23, 24]
    exp = [42, 43, 44, 45, 46, 47, 48, 49]

    atividades = ['W', 'W\_U', 'W\_D', 'SIT', 'STAND', 'LAY',
                  'S\_SIT', 'S\_STAND', 'SIT\_lay', 'L\_SIT', 'STAND\_lay', 'L\_STAND']

    # Frequência de amostragem definida no site referenciado no enunciado
    frequency = 50

    # Dinâmicos
    w_x = [[], [], [], [], [], [], [], []]
    w_y = [[], [], [], [], [], [], [], []]
    w_z = [[], [], [], [], [], [], [], []]
    w_u_x = [[], [], [], [], [], [], [], []]
    w_u_y = [[], [], [], [], [], [], [], []]
    w_u_z = [[], [], [], [], [], [], [], []]
    w_d_x = [[], [], [], [], [], [], [], []]
    w_d_y = [[], [], [], [], [], [], [], []]
    w_d_z = [[], [], [], [], [], [], [], []]

    # Estáticos
    sit_x = [[], [], [], [], [], [], [], []]
    sit_y = [[], [], [], [], [], [], [], []]
    sit_z = [[], [], [], [], [], [], [], []]
    stand_x = [[], [], [], [], [], [], [], []]
    stand_y = [[], [], [], [], [], [], [], []]
    stand_z = [[], [], [], [], [], [], [], []]
    lay_x = [[], [], [], [], [], [], [], []]
    lay_y = [[], [], [], [], [], [], [], []]
    lay_z = [[], [], [], [], [], [], [], []]

    # Transição
    stand_sit_x = [[], [], [], [], [], [], [], []]
    stand_sit_y = [[], [], [], [], [], [], [], []]
    stand_sit_z = [[], [], [], [], [], [], [], []]
    sit_stand_x = [[], [], [], [], [], [], [], []]
    sit_stand_y = [[], [], [], [], [], [], [], []]
    sit_stand_z = [[], [], [], [], [], [], [], []]
    sit_lay_x = [[], [], [], [], [], [], [], []]
    sit_lay_y = [[], [], [], [], [], [], [], []]
    sit_lay_z = [[], [], [], [], [], [], [], []]
    lay_sit_x = [[], [], [], [], [], [], [], []]
    lay_sit_y = [[], [], [], [], [], [], [], []]
    lay_sit_z = [[], [], [], [], [], [], [], []]
    stand_lay_x = [[], [], [], [], [], [], [], []]
    stand_lay_y = [[], [], [], [], [], [], [], []]
    stand_lay_z = [[], [], [], [], [], [], [], []]
    lay_stand_x = [[], [], [], [], [], [], [], []]
    lay_stand_y = [[], [], [], [], [], [], [], []]
    lay_stand_z = [[], [], [], [], [], [], [], []]

    array_activities = [w_x, w_y, w_z, w_u_x, w_u_y, w_u_z,
                    w_d_x, w_d_y, w_d_z, sit_x, sit_y, sit_z,
                    stand_x, stand_y, stand_z, lay_x, lay_y, lay_z, stand_sit_x, stand_sit_y,
                    stand_sit_z, sit_stand_x, sit_stand_y, sit_stand_z, sit_lay_x, sit_lay_y, sit_lay_z, lay_sit_x,
                    lay_sit_y, lay_sit_z, stand_lay_x, stand_lay_y, stand_lay_z, lay_stand_x, lay_stand_y,
                    lay_stand_z]

    max_mag_x = []
    max_mag_y = []
    max_mag_z = []
    min_mag_x = []
    min_mag_y = []
    min_mag_z = []

    for i in range(8):
        arr = open_users_data(exp[i], user[i // 2])
        # Arrays com dados de cada sensor
        x = []
        y = []
        z = []

        # Array com os valores de tempo, em segundos, de cada valor da experiência
        tempo = []

        colors = ["green", "blue", "red", "lightgreen", "yellow", "brown",
                  "darkred", "cyan", "orange", "olive", "gray", "pink"]

        # Guarda os valores de tempo e os dados de cada sensor nos seus respetivos arrays
        for j in arr:
            tempo.append(len(x) / (frequency * 60))
            x.append(j[0])
            y.append(j[1])
            z.append(j[2])

        # plot_signals_activities(exp[i], user[i//2])
        activities_separation(exp[i], user[i // 2], i)
        # plot_DFT_activities(w_x, w_y, w_z, i, exp[i], user[i//2], atividades[0])


    for j in range(0, len(array_activities), 3):
        for i in range(8):
            calculate_max_mag(array_activities[j][i], array_activities[j+1][i], array_activities[j+2][i])
        print(atividades[j//3] + " MAX")
        print(max(max_mag_x), end=" | ")
        print(max(max_mag_y), end=" | ")
        print(max(max_mag_z))

        max_mag_x.clear()
        max_mag_y.clear()
        max_mag_z.clear()

    print("----------------------------------------------------------------")
    for j in range(0, len(array_activities), 3):
        for i in range(8):
            calculate_min_mag(array_activities[j][i], array_activities[j+1][i], array_activities[j+2][i])
        print(atividades[j//3] + " MIN")
        print(min(min_mag_x), end=" | ")
        print(min(min_mag_y), end=" | ")
        print(min(min_mag_z))

        min_mag_x.clear()
        min_mag_y.clear()
        min_mag_z.clear()

    
    eixo_x = steps_per_min(w_x, w_u_x, w_d_x)
    eixo_y = steps_per_min(w_y, w_u_y, w_d_y)
    eixo_z = steps_per_min(w_z, w_u_z, w_d_z)

    print("%12s %4s %12s %4s %12s" % ("Eixo X", "|", "Eixo Y", "|", "Eixo Z"))
    for i in range(2):
        print("%s | %s | %s" % (eixo_x[i], eixo_y[i], eixo_z[i]))
    print("%s  | %s  | %s" % (eixo_x[2], eixo_y[2], eixo_z[2]))

    plot_windows(w_x[0])

    diferenciate_activities()
    """
    arr = open_users_data(exp[0], user[0])
    plot_STFT(arr, exp[0], user[0])
    """