import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix
)

print("Carregando e transformando a base California Housing...")
data = fetch_california_housing(as_frame=True)
df = data.frame

median_val = df['MedHouseVal'].median()
df['Target'] = (df['MedHouseVal'] > median_val).astype(int)

X = df.drop(columns=['MedHouseVal', 'Target'])
y = df['Target']

print(f"Base de dados: {X.shape}")
print(f"Classes (0/1): {y.value_counts(normalize=True).to_dict()}")
print("-" * 30)

print("Dividindo os dados (70% Treino, 15% Validação, 15% Teste)...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results_table = {}

SCORING_METRIC = 'f1' 

print("\n--- Iniciando Avaliação dos Modelos ---")

for name, model in models.items():
    print(f"\n--- Processando: {name} ---")
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    y_pred_train = model.predict(X_train_scaled)
    f1_train = f1_score(y_train, y_pred_train)
    
    y_pred_val = model.predict(X_val_scaled)
    f1_val = f1_score(y_val, y_pred_val)
    
    print(f"Tempo de Treino: {train_time:.4f}s")
    print(f"F1-Score (Treino):     {f1_train:.4f}")
    print(f"F1-Score (Validação):  {f1_val:.4f}")

    if f1_train > (f1_val + 0.1): 
        print("Status: Provável Overfitting (Treino muito melhor que Validação)")
    elif f1_train < 0.6 and f1_val < 0.6:
        print("Status: Provável Underfitting (Ambos os scores baixos)")
    else:
        print("Status: Bom equilíbrio")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=cv, scoring=SCORING_METRIC, n_jobs=-1
    )
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"Validação Cruzada (k=5, F1): Média = {cv_mean:.4f} | Desvio Padrão = {cv_std:.4f}")
    if cv_std > 0.05:
        print("Status CV: Modelo instável (alto desvio padrão)")
    else:
        print("Status CV: Modelo estável (baixo desvio padrão)")
        
    y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
    auc_val = roc_auc_score(y_val, y_prob_val)
    print(f"AUC (Validação): {auc_val:.4f}")

    y_pred_test = model.predict(X_test_scaled)
    f1_test = f1_score(y_test, y_pred_test)
    
    np.random.seed(42)
    X_test_noisy = X_test_scaled + np.random.normal(0, 0.1, X_test_scaled.shape)
    y_pred_noisy = model.predict(X_test_noisy)
    f1_test_noisy = f1_score(y_test, y_pred_noisy)
    
    print(f"F1-Score (Teste Puro):         {f1_test:.4f}")
    print(f"F1-Score (Teste Generalização): {f1_test_noisy:.4f}")
    
    if (f1_test - f1_test_noisy) > 0.05:
        print("Status Generalização: Frágil (desempenho caiu com ruído)")
    else:
        print("Status Generalização: Robusto (desempenho se manteve)")
        
    results_table[name] = {
        "F1 (Treino)": f1_train,
        "F1 (Val)": f1_val,
        "CV Média (F1)": cv_mean,
        "CV Desv. Padrão": cv_std,
        "AUC (Val)": auc_val,
        "F1 (Teste)": f1_test,
        "F1 (Generalização)": f1_test_noisy
    }

plt.figure(figsize=(10, 7))
for name, model in models.items():
    y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_prob_val)
    auc_val = roc_auc_score(y_val, y_prob_val)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC (Conjunto de Validação)')
plt.legend()
plt.show()

print("\n" + "="*50)
print("=== TABELA COMPARATIVA FINAL (Métrica: F1-Score) ===")
print("="*50)
df_results = pd.DataFrame(results_table).T
print(df_results.to_markdown(floatfmt=".4f"))

print("\n" + "="*50)
print("=== CONCLUSÃO (Responda com base na tabela) ===")
print("="*50)
print("""
**1. Equilíbrio (Overfitting/Underfitting):**
- Qual modelo teve o F1(Treino) e F1(Val) mais próximos e mais altos?
- Algum modelo mostrou F1(Treino) muito maior que F1(Val), indicando overfitting?

**2. Estabilidade:**
- Qual modelo teve o menor Desvio Padrão na Validação Cruzada (CV Desv. Padrão)?
- Um Desvio Padrão baixo sugere que o modelo é estável e seu desempenho não varia
muito com diferentes fatias dos dados.

**3. Capacidade de Separação (AUC):**
- Qual modelo obteve o maior AUC?
- Valores próximos de 0.9 são excelentes, 0.8 são bons.

**4. Generalização:**
- Qual modelo teve a menor queda de desempenho entre 'F1 (Teste)' e 
'F1 (Generalização)'?
- Um modelo que generaliza bem mantém seu desempenho mesmo com 
pequenas variações nos dados.

**5. Veredito Final:**
- Com base em todos os critérios, qual modelo você escolheria e por quê?
""")