# ==========================================
# ANÁLISE DE SENTIMENTOS - PARTE IV
# ==========================================

import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# GARANTINDO RECURSOS NLTK
# ==========================================
try:
    stop_words = set(stopwords.words("portuguese"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("portuguese"))

try:
    stemmer = RSLPStemmer()
except LookupError:
    nltk.download("rslp")
    stemmer = RSLPStemmer()

# ==========================================
# FUNÇÃO DE PRÉ-PROCESSAMENTO
# ==========================================
def preprocess(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palavras = texto.split()
    palavras = [stemmer.stem(p) for p in palavras if p not in stop_words]
    return " ".join(palavras)

# ==========================================
# DATASET
# ==========================================
data = {
    "frase": [
        "Eu amei este filme, foi maravilhoso!",
        "Que dia incrível, estou muito feliz!",
        "O atendimento foi péssimo e demorado.",
        "Não gostei do produto, muito ruim.",
        "Adorei a comida, estava deliciosa!",
        "Esse serviço é horrível, não recomendo.",
        "Estou muito satisfeito com o resultado.",
        "O filme foi uma perda de tempo.",
        "Experiência fantástica, voltarei com certeza!",
        "Detestei, foi uma experiência ruim.",
        "O livro é excelente e inspirador!",
        "Horrível, nunca mais compro neste lugar.",
        "A equipe foi muito atenciosa e prestativa.",
        "Péssimo atendimento, fiquei decepcionado.",
        "Maravilhoso, superou minhas expectativas!"
    ],
    "sentimento": [1,1,0,0,1,0,1,0,1,0,1,0,1,0,1]
}

df = pd.DataFrame(data)
df["frase_limpa"] = df["frase"].apply(preprocess)

# ==========================================
# TREINO E TESTE
# ==========================================
X = df["frase_limpa"]
y = df["sentimento"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ==========================================
# MODELOS
# ==========================================
regressao_logistica = LogisticRegression()
regressao_logistica.fit(X_train_tfidf, y_train)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)

# ==========================================
# MATRIZ DE CONFUSÃO + ANÁLISE DE ERROS
# ==========================================
def analisar_erros(nome, modelo):
    print(f"\n=== {nome} ===")
    y_pred = modelo.predict(X_test_tfidf)
    print("Acurácia:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negativo", "Positivo"]
    )
    disp.plot()
    plt.title(f"Matriz de Confusão - {nome}")
    plt.show()

    erros = pd.DataFrame({
        "frase": X_test.values,
        "real": y_test.values,
        "predito": y_pred
    })

    print("\nFalsos Positivos:")
    print(erros[(erros.real == 0) & (erros.predito == 1)])

    print("\nFalsos Negativos:")
    print(erros[(erros.real == 1) & (erros.predito == 0)])

# ==========================================
# PROBABILIDADE + PESOS
# ==========================================
def prever_com_confianca(modelo, frases):
    frases_proc = [preprocess(f) for f in frases]
    X_tfidf = tfidf.transform(frases_proc)

    probs = modelo.predict_proba(X_tfidf)
    preds = modelo.predict(X_tfidf)

    vocab = tfidf.get_feature_names_out()
    pesos = modelo.coef_[0]

    for i, frase in enumerate(frases):
        classe = "Positivo" if preds[i] == 1 else "Negativo"
        confianca = probs[i][preds[i]] * 100

        print(f"\nFrase: {frase}")
        print(f"Sentimento: {classe} ({confianca:.1f}%)")

        indices = X_tfidf[i].nonzero()[1]
        palavras = [(vocab[j], pesos[j]) for j in indices]
        palavras = sorted(palavras, key=lambda x: abs(x[1]), reverse=True)

        print("Palavras relevantes:", palavras[:5])

# ==========================================
# VISUALIZAÇÃO DOS PESOS
# ==========================================
def plot_pesos(modelo, tfidf, top_n=15):
    vocab = tfidf.get_feature_names_out()
    pesos = modelo.coef_[0]

    indices = np.argsort(np.abs(pesos))[-top_n:]
    palavras = [vocab[i] for i in indices]
    valores = pesos[indices]

    plt.figure(figsize=(10,5))
    plt.barh(palavras, valores)
    plt.axvline(0, color="black")
    plt.title("Palavras mais importantes (Regressão Logística)")
    plt.xlabel("Peso")
    plt.show()

# ==========================================
# EXECUÇÃO
# ==========================================
analisar_erros("Regressão Logística", regressao_logistica)
analisar_erros("Naive Bayes", naive_bayes)

novas_frases = [
    "O produto é excelente e chegou rápido",
    "Estou decepcionado com o serviço",
    "Não foi ruim, mas poderia ser melhor"
]

print("\n=== Simulação de Uso Real ===")
prever_com_confianca(regressao_logistica, novas_frases)

plot_pesos(regressao_logistica, tfidf)
