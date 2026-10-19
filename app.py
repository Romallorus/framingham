import io
import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ---------- КОНСТАНТЫ ----------
MODEL_PATH = "model.cbm"
MEDIANS_PATH = "medians.json"

FEATURES = [
    "male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
    "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP",
    "diaBP", "BMI", "heartRate", "glucose"
]

FEATURE_LABELS = {
    "male": "Пол (0 - Женский, 1 - Мужской)",
    "age": "Возраст, лет",
    "education": "Образование (1-4)",
    "currentSmoker": "Курит в настоящее время (0/1)",
    "cigsPerDay": "Сигарет в день",
    "BPMeds": "Принимает лекарства от давления (0/1)",
    "prevalentStroke": "Был инсульт (0/1)",
    "prevalentHyp": "Гипертония (0/1)",
    "diabetes": "Диабет (0/1)",
    "totChol": "Общий холестерин, мг/дл",
    "sysBP": "Систолическое давление",
    "diaBP": "Диастолическое давление",
    "BMI": "Индекс массы тела (ИМТ)",
    "heartRate": "Пульс (уд/мин)",
    "glucose": "Уровень глюкозы, мг/дл"
}

FEATURE_DESCRIPTIONS = {
    "male": "Пол пациента. 0 — женщина, 1 — мужчина.",
    "age": "Возраст пациента в годах.",
    "education": "Уровень образования (1 — начальное, 2 — среднее, 3 — высшее, 4 — ученое).",
    "currentSmoker": "Курит ли пациент в настоящее время (0 — нет, 1 — да).",
    "cigsPerDay": "Среднее количество выкуриваемых сигарет в день.",
    "BPMeds": "Принимает ли пациент препараты от артериального давления (0 — нет, 1 — да).",
    "prevalentStroke": "Был ли у пациента инсульт в прошлом (0 — нет, 1 — да).",
    "prevalentHyp": "Есть ли у пациента гипертония (0 — нет, 1 — да).",
    "diabetes": "Есть ли у пациента диабет (0 — нет, 1 — да).",
    "totChol": "Общий уровень холестерина в крови (мг/дл).",
    "sysBP": "Систолическое артериальное давление (верхнее число).",
    "diaBP": "Диастолическое артериальное давление (нижнее число).",
    "BMI": "Индекс массы тела (вес в кг / рост в метрах в квадрате).",
    "heartRate": "Частота сердечных сокращений (ударов в минуту).",
    "glucose": "Уровень глюкозы в крови (мг/дл)."
}

TEST_EXAMPLE = {
    "male": 1, "age": 50, "education": 2, "currentSmoker": 1, "cigsPerDay": 10,
    "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 1, "diabetes": 0,
    "totChol": 240, "sysBP": 140, "diaBP": 90, "BMI": 28.5, "heartRate": 75, "glucose": 85
}


# ---------- ЗАГРУЗКА ----------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not CATBOOST_AVAILABLE:
        raise RuntimeError("Библиотека catboost не установлена.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл модели '{path}' не найден.")
    model = CatBoostClassifier()
    model.load_model(path)
    return model


@st.cache_data(show_spinner=False)
def load_medians(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ---------- ИНИЦИАЛИЗАЦИЯ И УТИЛИТЫ ----------
def init_state():
    if "inputs" not in st.session_state:
        # Берем медианы как дефолтные значения
        medians = load_medians(MEDIANS_PATH)
        st.session_state.inputs = {feat: medians.get(feat, TEST_EXAMPLE.get(feat, 0)) for feat in FEATURES}
    if "history" not in st.session_state:
        st.session_state.history = []


def reset_form():
    medians = load_medians(MEDIANS_PATH)
    st.session_state.inputs = {feat: medians.get(feat, TEST_EXAMPLE.get(feat, 0)) for feat in FEATURES}


def load_test_example():
    st.session_state.inputs = TEST_EXAMPLE.copy()


def restore_from_history(idx):
    # Восстанавливаем данные из истории
    st.session_state.inputs = st.session_state.history[idx]["inputs"].copy()
    st.toast(f"Данные пациента '{st.session_state.history[idx]['name']}' восстановлены!")


# ---------- РУЧНОЙ ВВОД ----------
def manual_input_form():
    st.subheader("Введите клинические показатели")

    patient_name = st.text_input("Имя пациента (для истории)", value="Пациент А", key="input_name")

    vals = st.session_state.inputs
    cols = st.columns(2)

    with cols[0]:
        vals["male"] = st.selectbox(FEATURE_LABELS["male"], [0, 1], index=[0, 1].index(int(vals["male"])),
                                    key="in_male")
        vals["age"] = st.number_input(FEATURE_LABELS["age"], 20, 100, int(vals["age"]), key="in_age")
        vals["education"] = st.number_input(FEATURE_LABELS["education"], 1, 4, int(vals["education"]), step=1,
                                            key="in_edu")
        vals["currentSmoker"] = st.selectbox(FEATURE_LABELS["currentSmoker"], [0, 1],
                                             index=[0, 1].index(int(vals["currentSmoker"])), key="in_smoker")
        vals["cigsPerDay"] = st.number_input(FEATURE_LABELS["cigsPerDay"], 0, 60, int(vals["cigsPerDay"]),
                                             key="in_cigs")
        vals["BPMeds"] = st.selectbox(FEATURE_LABELS["BPMeds"], [0, 1], index=[0, 1].index(int(vals["BPMeds"])),
                                      key="in_bpmeds")
        vals["prevalentStroke"] = st.selectbox(FEATURE_LABELS["prevalentStroke"], [0, 1],
                                               index=[0, 1].index(int(vals["prevalentStroke"])), key="in_stroke")
        vals["prevalentHyp"] = st.selectbox(FEATURE_LABELS["prevalentHyp"], [0, 1],
                                            index=[0, 1].index(int(vals["prevalentHyp"])), key="in_hyp")

    with cols[1]:
        vals["diabetes"] = st.selectbox(FEATURE_LABELS["diabetes"], [0, 1], index=[0, 1].index(int(vals["diabetes"])),
                                        key="in_diab")
        vals["totChol"] = st.number_input(FEATURE_LABELS["totChol"], 100, 600, int(vals["totChol"]), key="in_chol")
        vals["sysBP"] = st.number_input(FEATURE_LABELS["sysBP"], 80, 220, int(vals["sysBP"]), key="in_sysbp")
        vals["diaBP"] = st.number_input(FEATURE_LABELS["diaBP"], 40, 140, int(vals["diaBP"]), key="in_diabp")
        vals["BMI"] = st.number_input(FEATURE_LABELS["BMI"], 15.0, 50.0, float(vals["BMI"]), step=0.1, key="in_bmi")
        vals["heartRate"] = st.number_input(FEATURE_LABELS["heartRate"], 40, 200, int(vals["heartRate"]), key="in_hr")
        vals["glucose"] = st.number_input(FEATURE_LABELS["glucose"], 40, 400, int(vals["glucose"]), key="in_gluc")

    st.session_state.inputs = vals
    return pd.DataFrame([vals], columns=FEATURES), patient_name


# ---------- CSV И ПРЕДОБРАБОТКА ----------
def csv_input_block():
    st.subheader("Загрузите CSV-файл")
    st.caption("Один пациент — одна строка. Колонки должны совпадать с признаками модели (15 признаков).")

    template = pd.DataFrame([TEST_EXAMPLE], columns=FEATURES)
    st.download_button(
        "📥 Скачать шаблон CSV (без таргета)",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="framingham_template.csv", mime="text/csv",
        help="Скачайте пример файла для проверки модели без таргета."
    )

    check_model = st.checkbox("Проверить работу модели (загрузите CSV с колонкой TenYearCHD)",
                              help="Модель предскажет риски и сравнив с реальными ответами, покажет метрики точности.")

    if check_model:
        st.info(
            "⚠️ **Внимание!** Для проверки метрик ваш CSV-файл ОБЯЗАТЕЛЬНО должен содержать колонку `TenYearCHD` с правильными ответами (0 или 1).")

    uploaded = st.file_uploader("CSV с клиническими данными", type=["csv"])
    if uploaded is None:
        return None, None, None, False

    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Не удалось прочитать файл. Проверьте, что это корректный CSV.")
        return None, None, None, False

    # Обработка имени файла
    name_col = next((col for col in df.columns if col.lower() in ["name", "имя"]), None)
    if name_col:
        patient_names = df[name_col].astype(str).tolist()
        df = df.drop(columns=[name_col])
    else:
        patient_names = [f"Файл: {uploaded.name}"] * len(df)

    # 1. УДАЛЯЕМ ДУБЛИКАТЫ
    rows_before = len(df)
    df = df.drop_duplicates()
    if len(df) < rows_before:
        st.warning(f"Удалено дубликатов: {rows_before - len(df)}")

    # Проверка таргета
    target_data = None
    if check_model:
        if "TenYearCHD" not in df.columns:
            st.error("Ошибка: Вы поставили галочку проверки модели, но в файле нет колонки 'TenYearCHD'.")
            return None, None, None, False
        target_data = df["TenYearCHD"]
        df = df.drop(columns=["TenYearCHD"])

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"В файле не хватает колонок: {', '.join(missing)}")
        return None, None, None, False

    df = df[FEATURES]

    # 2. ОБРАБОТКА ПРОПУСКОВ (NaN)
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        st.warning(f"Обнаружены пропуски: {nan_count} ячеек.")
        strategy = st.radio(
            "Что делать с пропусками (NaN)?",
            ["Заполнить медианами из файла medians.json", "Удалить строки с пропусками"],
            key="nan_strategy"
        )

        if strategy == "Заполнить медианами из файла medians.json":
            medians = load_medians(MEDIANS_PATH)
            df = df.fillna(medians)
        else:
            idx_before = df.index
            df = df.dropna().reset_index(drop=True)
            if len(df) < len(idx_before):
                st.info(f"Удалено строк с пропусками: {len(idx_before) - len(df)}")
                if target_data is not None:
                    target_data = target_data.loc[df.index]  # Синхронизируем таргет

    if df.empty:
        st.error("После очистки не осталось данных.")
        return None, None, None, False

    st.success(f"Готово к анализу строк: {len(df)}")
    st.dataframe(df, use_container_width=True)

    return df, patient_names[:len(df)], target_data, check_model


# ---------- ПРЕДСКАЗАНИЕ И МЕТРИКИ ----------
def predict_and_render(model, df, names, target_data, check_model):
    try:
        probs = model.predict_proba(df)
        preds = model.predict(df).astype(int).ravel()
    except Exception as exc:
        st.error(f"Ошибка при вызове модели: {exc}")
        return

    st.subheader("Результаты анализа")

    # Если режим Проверки модели (с таргетом)
    if check_model and target_data is not None:
        st.info(
            "На вход подаются данные с X, y. Проверка предсказаний полезна, если нам нужно понять, устарела ли модель, стоит ли её переобучать.")

        col1, col2 = st.columns(2)
        with col1:
            st.text("Classification Report:")
            report = classification_report(target_data, preds, zero_division=0)
            st.code(report, language="text")
        with col2:
            st.metric("ROC-AUC", f"{roc_auc_score(target_data, probs[:, 1]):.4f}")

            # Матрица ошибок
            cm = confusion_matrix(target_data, preds)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            st.write("**Матрица ошибок (Confusion Matrix):**")
            st.write(f"✅ True Positive (TP): **{tp}** | ❌ False Positive (FP): **{fp}**")
            st.write(f"❌ False Negative (FN): **{fn}** | ✅ True Negative (TN): **{tn}**")

    # Сохранение в историю
    history_record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": names[0] if isinstance(names, list) else names,
        "inputs": df.iloc[0].to_dict() if len(df) == 1 else {},
        "verdict": "Анализ завершен"
    }

    if len(df) == 1:
        p_disease = float(probs[0][1])

        # Логика вердикта
        if p_disease < 0.4:
            verdict = "✅ Риск ИБС не обнаружен"
            history_record["verdict"] = verdict
            st.success(f"### {verdict}")
        elif 0.4 <= p_disease <= 0.6:
            verdict = "⚠️ Серая зона: Есть риски, но небольшие. Рекомендуется ежегодное посещение врача."
            history_record["verdict"] = "Серая зона"
            st.warning(f"### {verdict}")
        else:
            verdict = "🚨 Повышенный риск ИБС в ближайшие 10 лет"
            history_record["verdict"] = "Высокий риск"
            st.error(f"### {verdict}")

        st.metric("Вероятность наличия ИБС", f"{p_disease * 100:.1f} %")

        # Сохраняем в историю и выводим Plotly
        if names[0] not in ["Файл: ", ""]:
            st.session_state.history.append(history_record)

        # Plotly Вероятности
        proba_df = pd.DataFrame({"Класс": ["Нет ИБС", "ИБС"], "Вероятность": [probs[0][0], probs[0][1]]})
        fig = px.bar(proba_df, x="Класс", y="Вероятность", color="Класс",
                     title="Вероятность классов", color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Пакетный CSV
        results = df.copy()
        results.insert(0, "Имя", names)
        results["Вероятность ИБС"] = probs[:, 1].round(3)
        results["Предсказание"] = preds

        # Count Plot
        count_df = results["Предсказание"].value_counts().reset_index()
        count_df.columns = ["Класс", "Количество"]
        count_df["Класс"] = count_df["Класс"].map({0: "Здоров (0)", 1: "ИБС (1)"})
        fig = px.bar(count_df, x="Класс", y="Количество", color="Класс",
                     title="Распределение предсказанных классов", color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(results, use_container_width=True)

        # Кнопка скачать CSV с результатами
        st.download_button("⬇️ Скачать результаты (CSV)",
                           data=results.to_csv(index=False).encode('utf-8'),
                           file_name="predictions.csv", mime="text/csv")

    # Важность признаков
    render_feature_importance(model)


@st.cache_data(show_spinner=False)
def get_feature_importance(_model):
    fi = _model.get_feature_importance()
    return pd.DataFrame({"Признак": FEATURES, "Важность": fi}).sort_values("Важность", ascending=False)


def render_feature_importance(model):
    st.subheader("Вклад признаков модели")
    try:
        fi_df = get_feature_importance(model)
        fig = px.bar(fi_df, x="Важность", y="Признак", orientation='h', title="Важность признаков")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.info(f"Важность признаков недоступна: {exc}")


# ---------- ГЛАВНАЯ ФУНКЦИЯ ----------
def main():
    st.set_page_config(page_title="Прогноз ИБС (Framingham)", page_icon="🫀", layout="wide")
    init_state()

    st.title("🫀 Прогноз ИБС на 10 лет (Framingham)")
    st.caption("Учебное демо. Не является медицинским заключением.")

    model = None
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        st.warning(str(exc))

    # --- БОКОВАЯ ПАНЕЛЬ ---
    with st.sidebar:
        st.header("Управление")
        st.button("🧹 Очистить форму (сброс цифр)", on_click=reset_form, use_container_width=True,
                  help="Сбрасывает все поля ввода в значения по умолчанию.")
        st.button("🧪 Тестовый пример", on_click=load_test_example, use_container_width=True,
                  help="Заполняет форму примером пациента с известными показателями.")

        st.divider()

        st.subheader("📜 История запросов")
        if not st.session_state.history:
            st.caption("История пока пуста.")
        else:
            for i, item in enumerate(reversed(st.session_state.history)):
                real_idx = len(st.session_state.history) - 1 - i
                st.markdown(f"**[{item['time']}] {item['name']}**")
                st.caption(f"Результат: {item['verdict']}")
                if item['inputs']:  # Если данные были введены вручную
                    if st.button(f"🔄 Воспроизвести", key=f"restore_{real_idx}", use_container_width=True):
                        restore_from_history(real_idx)
                        st.rerun()
                st.divider()

            if st.button("🗑️ Очистить историю", use_container_width=True, help="Удаляет все сохраненные результаты."):
                st.session_state.history = []
                st.rerun()

        with st.expander("📖 Справка по показателям"):
            for feat, desc in FEATURE_DESCRIPTIONS.items():
                st.markdown(f"**{FEATURE_LABELS[feat]}:** {desc}")

    # --- ОСНОВНОЙ ЭКРАН ---
    mode = st.radio("Способ ввода данных:", ["Ввести вручную", "Загрузить CSV-файл"], horizontal=True, key="radio_mode")

    df_input, input_names, target_data, check_model = None, "", None, False

    if mode == "Ввести вручную":
        df_input, input_names = manual_input_form()
        check_model = False
    else:
        df_input, input_names, target_data, check_model = csv_input_block()

    st.divider()

    if st.button("🔍 Запустить анализ", type="primary", use_container_width=True, key="btn_predict"):
        if model is None:
            st.error("Модель не загружена.")
        elif df_input is None:
            st.error("Сначала введите данные или загрузите CSV.")
        else:
            try:
                predict_and_render(model, df_input, input_names, target_data, check_model)
            except Exception as exc:
                st.error(f"Непредвиденная ошибка: {exc}")


if __name__ == "__main__":
    main()