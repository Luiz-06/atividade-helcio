import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils import resample 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA

# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer

def plotar_matriz_confusao(y_real, y_pred, titulo):
    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Não-Fraude (0)', 'Fraude (1)'],
                yticklabels=['Não-Fraude (0)', 'Fraude (1)'])
    plt.title(f'Matriz de Confusão - {titulo}')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.show()

print("Carregando Dados (Credit Card Fraud)...")
try:
    df = pd.read_csv(r"C:\caminho\para\creditcard.csv") 
except FileNotFoundError:
    print("ERRO: Arquivo 'creditcard.csv' não encontrado. Ajuste o caminho.")
    df = pd.DataFrame() 

if not df.empty:
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    print("CreditCard:", X.shape, "| Fraudes:", y.sum(), "| Não-Fraudes:", (y==0).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_samples_tuning = int(X_train_scaled.shape[0] * 0.3) 
    
    X_train_sample, y_train_sample = resample(
        X_train_scaled, y_train, 
        n_samples=n_samples_tuning, 
        random_state=42, 
        stratify=y_train
    )
    
    print(f"Usando {X_train_sample.shape[0]} amostras para o Tuning (de {X_train_scaled.shape[0]} totais de treino).")

SCORING_METRIC = 'recall' 

models_and_params = {
    "LogisticRegression": {
        "model": LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
        "params": {
            "penalty": ['l1', 'l2'],
            "C": [0.01, 0.1, 1, 10] 
        }
    },
    
    "SVC": {
        "model": SVC(random_state=42, probability=True), 
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ['linear', 'rbf'],
            "gamma": ['scale', 0.1]
        }
    },
    
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    }
}

results = [] 

for name, config in models_and_params.items():
    print(f"\n--- Processando: {name} ---")
    model_default = config["model"]
    
    print("Treinando modelo Padrão...")
    start_time = time.time()
    model_default.fit(X_train_scaled, y_train)
    y_pred_default = model_default.predict(X_test_scaled)
    time_default = time.time() - start_time
    
    results.append({
        "Base": "CreditCard",
        "Modelo": name,
        "Tuning": "Sem Tuning",
        "Acurácia": accuracy_score(y_test, y_pred_default),
        "Precisão": precision_score(y_test, y_pred_default),
        "Recall": recall_score(y_test, y_pred_default),
        "F1-Score": f1_score(y_test, y_pred_default),
        "ROC-AUC": roc_auc_score(y_test, model_default.predict_proba(X_test_scaled)[:, 1]),
        "Tempo (s)": time_default
    })
    
    print(f"Iniciando RandomizedSearchCV (n_iter=10, cv=5)...")
    rs = RandomizedSearchCV(
        estimator=config["model"],
        param_distributions=config["params"],
        n_iter=10, 
        cv=3,      
        scoring=SCORING_METRIC,
        n_jobs=-1, 
        random_state=42
    )
    
    start_time = time.time()
    rs.fit(X_train_sample, y_train_sample)
    time_random = time.time() - start_time
    
    best_model_rs = rs.best_estimator_
    y_pred_rs = best_model_rs.predict(X_test_scaled)
    
    results.append({
        "Base": "CreditCard",
        "Modelo": name,
        "Tuning": "RandomizedSearch",
        "Acurácia": accuracy_score(y_test, y_pred_rs),
        "Precisão": precision_score(y_test, y_pred_rs),
        "Recall": recall_score(y_test, y_pred_rs),
        "F1-Score": f1_score(y_test, y_pred_rs),
        "ROC-AUC": roc_auc_score(y_test, best_model_rs.predict_proba(X_test_scaled)[:, 1]),
        "Tempo (s)": time_random
    })
    
    print(f"Melhores params (Randomized): {rs.best_params_}")


df_results = pd.DataFrame(results)
print("\n=== Tabela Comparativa (Base: CreditCard) ===")
print(df_results.to_markdown(index=False, floatfmt=".4f"))

print("\n=== Análise Breve (CreditCard) ===")
# [Seu texto de análise aqui, 5-8 linhas] 

print("\n--- Plotando Fronteiras de Decisão (com PCA) ---")