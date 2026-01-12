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
    }.get(c, np.array([1,1,1,1.0]))


def generate_cell(mu, rng):
    return mu + rng.normal(0, 0.6, 4)


def label_cell(x):
    sig = lambda z: 1/(1+np.exp(-z))
    t,m,r,l = sig(x[0]), sig(x[1]), sig(x[2]), sig(0.7*x[3])
    stress = (abs(m-0.55)+abs(t-0.55)+max(0,0.6-r))/3
    s = np.array([
        0.7*(1-m) + 0.2*(1-r) + 0.3*stress,
        0.85*(1-l) + 0.15*stress + 0.1*r,
        0.6*r + 0.4*l,
        0.7*r + 0.3*stress
    ]) + np.random.normal(0,0.02,4)
    return (s > np.array([0.55,0.48,0.55,0.55])).astype(int)


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
    s = {"__g__": StandardScaler().fit(df[FEATS])}
    for r, sub in df.groupby("region_id"):
        s[r] = StandardScaler().fit(sub[FEATS])
    return s


@st.cache_resource
def train_action_net():
    df = pd.read_csv(CSV)
    regs = sorted(df["region_id"].dropna().unique())
    scalers = fit_scalers(df)
    rng = np.random.default_rng(1)
    cultures = ["Пшеница","Рис","Картофель","Кукуруза","Подсолнечник"]

    X,Y = [],[]
    for _ in range(12000):
        r = rng.choice(regs)
        c = rng.choice(cultures)
        sub = df[df.region_id==r]
        mu = sub[FEATS].mean().values
        sc = scalers.get(r, scalers["__g__"])
        x = sc.transform([generate_cell(mu,rng)])[0] * culture_mult(c)
        X.append(x)
        Y.append(label_cell(x))

    X,Y = np.array(X), np.array(Y)
    Xtr,Xte,Ytr,Yte = train_test_split(X,Y,test_size=0.2,random_state=1)

    m = build_action_net()
    m.fit(Xtr,Ytr,epochs=10,batch_size=256,verbose=0)
    p = (m.predict(Xte,verbose=0)>0.5).astype(int)

    return df, regs, scalers, m, {
        "hamming": hamming_loss(Yte,p),
        "f1": f1_score(Yte,p,average="micro")
    }


def tractor_route_snake(H, W):
    # оставляю твою функцию (может пригодиться)
    path = []
    for r in range(H):
        if r % 2 == 0:
            for c in range(W):
                path.append((r, c))
        else:
            for c in range(W - 1, -1, -1):
                path.append((r, c))
    return path


def grid_size(w,h,spm):
    H0,W0=int(h*spm),int(w*spm)
    s=max(1,int(np.ceil(max(H0/MAX_SIDE,W0/MAX_SIDE))))
    return max(1,H0//s),max(1,W0//s),s/spm


# ============================================================
# УМНАЯ "ЗМЕЙКА" + ВЫЕЗД ЗА ЗАПАСОМ СЛЕВА (c = -1)
# ============================================================

def manhattan_path_ext(a, b):
    """
    Кратчайший путь по "решётке" без препятствий.
    Разрешаем колонку -1 как "запасы слева от поля".
    """
    (r1,c1),(r2,c2) = a,b
    path=[(r1,c1)]
    r,c=r1,c1
    while r != r2:
        r += 1 if r2>r else -1
        path.append((r,c))
    while c != c2:
        c += 1 if c2>c else -1
        path.append((r,c))
    return path


def smart_snake_route_for_mask(mask2d):
    """
    Немного "умная змейка":
    - едем не по всему полю, а только по сегментам строк, где mask=True
    Возвращает маршрут внутри поля (r, c) где c в [0..W-1]
    """
    H,W = mask2d.shape
    if not mask2d.any():
        return [(0,0)]

    segments = []
    for r in range(H):
        cols = np.where(mask2d[r])[0]
        if len(cols)==0:
            continue
        segments.append((r, int(cols.min()), int(cols.max())))
    segments.sort(key=lambda x: x[0])

    path = [(0,0)]
    cur = (0,0)

    for i,(r,cmin,cmax) in enumerate(segments):
        left_to_right = (i % 2 == 0)
        entry = (r, cmin) if left_to_right else (r, cmax)

        # перейти к входу сегмента
        segmove = manhattan_path_ext(cur, entry)
        path.extend(segmove[1:])
        cur = entry

        # пройти по сегменту
        if left_to_right:
            for c in range(cmin, cmax+1):
                if (r,c) != cur:
                    path.append((r,c))
                    cur = (r,c)
        else:
            for c in range(cmax, cmin-1, -1):
                if (r,c) != cur:
                    path.append((r,c))
                    cur = (r,c)

    return path


def add_refills_to_route(base_route, work_mask, capacity, refill_left_col=-1):
    """
    base_route: маршрут по полю (r,c) с c>=0
    work_mask: (H,W) bool — где реально нужно выполнить эту операцию
    capacity: сколько "клеток-работ" можно выполнить на одном баке
    refill point: слева от поля -> (r, -1) на той же строке (быстро и реалистично)

    Возвращает:
      full_route: список (r,c) где c может быть -1
      refills: сколько раз заезжали пополняться
      done_count: сколько работ закрыто
    """
    H,W = work_mask.shape
    if capacity <= 0:
        capacity = 1

    # если вообще нечего делать — просто остаёмся как есть
    if not work_mask.any():
        return base_route, 0, 0

    done = work_mask.copy()
    fuel = capacity
    refills = 0
    full_route = [base_route[0]]

    cur = base_route[0]

    def go_to(point):
        nonlocal cur, full_route
        seg = manhattan_path_ext(cur, point)
        full_route.extend(seg[1:])
        cur = point

    # пройдём по базовому маршруту, выполняя работу там, где нужно
    done_count = 0
    for nxt in base_route[1:]:
        go_to(nxt)  # дойдём до следующей клетки

        r,c = cur
        # если на этой клетке есть работа — тратим ресурс
        if 0 <= c < W and done[r,c]:
            if fuel == 0:
                # едем пополниться слева (на той же строке), потом возвращаемся сюда
                go_to((r, refill_left_col))
                refills += 1
                fuel = capacity
                go_to((r, c))

            # выполняем работу
            done[r,c] = False
            fuel -= 1
            done_count += 1

            # если ещё осталось много работы, а fuel стал 0 — пополнимся заранее на конце строки?
            # (оставим просто как есть — пополнится при необходимости)

        # если всё сделали — можно остановиться раньше
        if done_count == int(work_mask.sum()):
            break

    return full_route, refills, done_count


def padded_mask_for_display(mask2d):
    """
    Добавляем слева 1 колонку (для "запасов" c=-1).
    """
    H,W = mask2d.shape
    pad = np.zeros((H, W+1), dtype=mask2d.dtype)
    pad[:, 1:] = mask2d
    return pad


def shift_route_for_display(route):
    """
    Сдвигаем координаты по x на +1:
    c=-1 -> x=0 (запасы)
    c=0  -> x=1 (первая колонка поля)
    """
    rr = [r for (r,c) in route]
    cc = [c+1 for (r,c) in route]
    return rr, cc


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(layout="wide")
st.title("Автоматизация сельскохозяйственных операций")

df,regions,scalers,action_model,metrics = train_action_net()

with st.expander("Метрики модели операций"):
    st.write(metrics)

c1,c2,c3 = st.columns(3)
w = c1.number_input("Ширина поля (м)",1,1_000_000,200,10)
h = c2.number_input("Длина поля (м)",1,1_000_000,200,10)
spm = c3.number_input("Посевов в 1 метре",1,200,1)

c4,c5 = st.columns(2)
reg = c4.selectbox("Регион", regions)
cul = c5.selectbox("Культура", ["Пшеница","Рис","Картофель","Кукуруза","Подсолнечник"])

H,W,cell = grid_size(float(w),float(h),int(spm))

st.subheader("Запасы (ёмкости) для операций")
cc1,cc2,cc3,cc4 = st.columns(4)
cap = [
    cc1.number_input(f"Ёмкость: {OPS_RU[0]} (клеток)", 1, 1_000_000, 300, 10),
    cc2.number_input(f"Ёмкость: {OPS_RU[1]} (клеток)", 1, 1_000_000, 220, 10),
    cc3.number_input(f"Ёмкость: {OPS_RU[2]} (клеток)", 1, 1_000_000, 180, 10),
    cc4.number_input(f"Ёмкость: {OPS_RU[3]} (клеток)", 1, 1_000_000, 260, 10),
]

st.caption("Здесь 1 'единица' запаса = обработка 1 клетки соответствующей операцией. Запасы находятся слева от поля (служебная полоса).")

if st.button("Сгенерировать"):
    rng = np.random.default_rng(2)
    sub = df[df.region_id==reg]
    mu = sub[FEATS].mean().values
    sc = scalers.get(reg,scalers["__g__"])

    raw = mu + rng.normal(0,0.6,(H,W,4))
    x = sc.transform(raw.reshape(-1,4)) * culture_mult(cul)
    act = (action_model.predict(x,verbose=0)>0.5).astype(int).reshape(H,W,4)
    mask_any = act.any(axis=-1)

    # карты операций
    fig1,axs = plt.subplots(2,2,figsize=(9,9))
    for i in range(4):
        axs.flat[i].imshow(act[...,i])
        axs.flat[i].set_title(OPS_RU[i])
        axs.flat[i].axis("off")
    st.pyplot(fig1,clear_figure=True)

    # отдельные маршруты по операциям
    st.subheader("Маршруты трактора по операциям (с учётом пополнения запасов слева)")

    tabs = st.tabs([OPS_RU[i] for i in range(4)])

    for op in range(4):
        with tabs[op]:
            op_mask = act[...,op].astype(bool)

            if not op_mask.any():
                st.info("Нет клеток для этой операции.")
                continue

            # "умная змейка" для этой операции
            base = smart_snake_route_for_mask(op_mask)

            # добавим выезды за запасом
            route, refills, done_count = add_refills_to_route(
                base_route=base,
                work_mask=op_mask,
                capacity=int(cap[op]),
                refill_left_col=-1
            )

            # визуализация: добавим колонку слева для запасов
            disp_mask = padded_mask_for_display(op_mask.astype(int))

            fig, ax = plt.subplots(figsize=(9,9))
            ax.imshow(disp_mask)

            rr, cc = shift_route_for_display(route)
            ax.plot(cc, rr, linewidth=3)
            ax.scatter([cc[0]],[rr[0]],s=120)                 # старт
            ax.scatter([0],[rr[0]],s=160, marker="s")         # "запасы" слева на строке старта (условно)
            ax.scatter([cc[-1]],[rr[-1]],s=120,marker="X")    # финиш

            ax.axis("off")
            st.pyplot(fig, clear_figure=True)

            route_len_cells = max(0, len(route)-1)
            st.caption(
                f"Операция: {OPS_RU[op]} | "
                f"Клеток работы: {int(op_mask.sum())} | "
                f"Сделано: {done_count} | "
                f"Пополнений: {refills} | "
                f"Длина маршрута (клеток): {route_len_cells} | "
                f"Клетка ~ {cell:.2f} м | "
                f"Оценка пробега ~ {route_len_cells * cell:.1f} м"
            )

    # общий слой "есть хоть какая-то работа" + общий базовый маршрут (опционально)
    st.subheader("Общая карта: где есть хотя бы одна операция")
    fig2,ax2 = plt.subplots(figsize=(9,9))
    ax2.imshow(mask_any)
    ax2.axis("off")
    st.pyplot(fig2, clear_figure=True)

    st.caption(f"Размер сетки: {H}×{W} | Клетка ~ {cell:.2f} м")
