import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
CSV = "dataset_russia_subjects_normalized.csv"
FEATS = ["t2m", "swvl1", "tp", "lai_hv"]
OPS_RU = ["Полив", "Удобрение", "Опрыскивание", "Культивация"]
MAX_SIDE = 220
def culture_mult(c):
    c = (c or "").lower()
    return {
        "пшеница": np.array([1,1,1,1.0]),
        "рис": np.array([1,0.8,1.1,1.0]),
        "картофель": np.array([1,1,1,1.1]),
        "кукуруза": np.array([1.1,1,1,1.0]),
        "подсолнечник": np.array([1,1.1,1,0.9])
    }.get(c, np.ones(4))
def generate_cell(mu, rng):
    return mu + rng.normal(0, 0.6, 4)
def label_cell(x):
    sig = lambda z: 1/(1+np.exp(-z))
    t,m,r,l = sig(x[0]), sig(x[1]), sig(x[2]), sig(0.7*x[3])
    stress = (abs(m-0.55)+abs(t-0.55)+max(0,0.6-r))/3

    s = np.array([
        0.7*(1-m) + 0.3*stress,
        0.85*(1-l) + 0.15*stress,
        0.6*r + 0.4*l,
        0.7*r + 0.3*stress
    ])

    return (s > 0.55).astype(int)
def build_action_net():
    m = Sequential([
        Dense(128, activation='relu', input_dim=4),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(4, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
    return m
def fit_scalers(df):
    scalers = {}
    for r, sub in df.groupby("region_id"):
        scalers[r] = StandardScaler().fit(sub[FEATS])
    return scalers
@st.cache_resource
def train_action_net():
    df = pd.read_csv(CSV)
    regions = sorted(df["region_id"].unique())
    scalers = fit_scalers(df)

    rng = np.random.default_rng(1)
    cultures = ["Пшеница","Рис","Картофель","Кукуруза","Подсолнечник"]

    X, Y = [], []

    for _ in range(12000):
        r = rng.choice(regions)
        c = rng.choice(cultures)

        sub = df[df.region_id == r]
        mu = sub[FEATS].mean().values

        x = scalers[r].transform([generate_cell(mu, rng)])[0]
        x *= culture_mult(c)

        X.append(x)
        Y.append(label_cell(x))

    X, Y = np.array(X), np.array(Y)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2)

    model = build_action_net()
    model.fit(Xtr, Ytr, epochs=10, batch_size=256, verbose=0)

    return df, regions, scalers, model
def build_tractor_net():
    m = Sequential([
        Dense(16, activation="relu", input_dim=2),
        Dense(8, activation="relu"),
        Dense(4, activation="softmax")
    ])
    m.compile(
        optimizer=Adam(0.001),
        loss="sparse_categorical_crossentropy"
    )
    return m
@st.cache_resource
def train_tractor_net():
    rng = np.random.default_rng(2)
    X, y = [], []

    for _ in range(6000):
        r, c = rng.integers(0, 20, size=2)
        X.append([r/20, c/20])

        if r % 2 == 0:
            y.append(3)  # вправо
        else:
            y.append(2)  # влево

    m = build_tractor_net()
    m.fit(np.array(X), np.array(y), epochs=3, batch_size=256, verbose=0)
    return m
DIRS = {
    0: (-1, 0),  # вверх
    1: (1, 0),   # вниз
    2: (0, -1),  # влево
    3: (0, 1)    # вправо
}
def tractor_recursive(r, c, field, model, path, visited):
    if (r, c) in visited:
        return
    if not (0 <= r < field.shape[0] and 0 <= c < field.shape[1]):
        return

    visited.add((r, c))
    path.append((r, c))

    x = np.array([[r/field.shape[0], c/field.shape[1]]])
    move = np.argmax(model.predict(x, verbose=0))

    dr, dc = DIRS[move]
    tractor_recursive(r+dr, c+dc, field, model, path, visited)
st.set_page_config(layout="wide")
st.title("Автоматизация сельского хозяйства")

df, regions, scalers, action_model = train_action_net()
tractor_model = train_tractor_net()

c1,c2,c3 = st.columns(3)
w = c1.number_input("Ширина поля (м)", 50, 1000, 200)
h = c2.number_input("Длина поля (м)", 50, 1000, 200)
spm = c3.number_input("Ячеек на метр", 1, 5, 1)

reg = st.selectbox("Регион", regions)
cul = st.selectbox("Культура", ["Пшеница","Рис","Картофель","Кукуруза","Подсолнечник"])
if st.button("Сгенерировать"):
    H, W = int(h*spm), int(w*spm)

    rng = np.random.default_rng(3)
    sub = df[df.region_id == reg]
    mu = sub[FEATS].mean().values

    raw = mu + rng.normal(0, 0.6, (H, W, 4))
    x = scalers[reg].transform(raw.reshape(-1,4))
    x *= culture_mult(cul)

    act = (action_model.predict(x) > 0.5).astype(int).reshape(H,W,4)
    mask = act.any(axis=-1)

    fig, axs = plt.subplots(2,2, figsize=(9,9))
    for i in range(4):
        axs.flat[i].imshow(act[:,:,i])
        axs.flat[i].set_title(OPS_RU[i])
        axs.flat[i].axis("off")
    st.pyplot(fig)

    path = []
    tractor_recursive(0, 0, mask, tractor_model, path, set())

    fig2, ax = plt.subplots(figsize=(8,8))
    ax.imshow(mask)
    ax.plot([p[1] for p in path], [p[0] for p in path], linewidth=2)
    ax.set_title("Рекурсивный маршрут трактора")
    ax.axis("off")
    st.pyplot(fig2)
