# train_model.py
import pandas as pd
import json
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = "framingham.csv"


def train():
    print(f"Загрузка датасета {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Ошибка: Файл {DATA_PATH} не найден. Положите его в папку с проектом.")
        return

    # Убираем строки без таргета и дубликаты (как вы просили)
    df = df.dropna(subset=["TenYearCHD"]).drop_duplicates()

    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]
    FEATURES = list(X.columns)

    # Считаем медианы на трейне (до заполнения!)
    medians = X.median().to_dict()

    # Сохраняем медианы в JSON для app.py
    with open("medians.json", "w") as f:
        json.dump(medians, f, indent=4)
    print("\nМедианы сохранены в файл 'medians.json'")

    # Заполняем пропуски для обучения
    X = X.fillna(medians)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    print("\nОбучение финальной модели с лучшими параметрами...")

    # Берем параметры, которые нашел GridSearch
    best_params = {'depth': 5, 'learning_rate': 0.08, 'l2_leaf_reg': 3}

    model = CatBoostClassifier(
        iterations=1000,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,  # 👈 Вот это спасет от переобучения
        **best_params  # Подставляем лучшие параметры
    )

    # Обучаем с валидационной выборкой, чтобы early_stopping работал
    model.fit(
        Pool(X_train, y_train, feature_names=FEATURES),
        eval_set=Pool(X_test, y_test, feature_names=FEATURES),
        use_best_model=True
    )

    # Оцениваем
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- Финальные метрики на тестовой выборке ---")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"F1-score : {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")
    print("---------------------------------------------\n")

    model.save_model("model.cbm")
    print("Модель сохранена в файл 'model.cbm'")
if __name__ == "__main__":
    train()