# ==============================
# 1. Importando bibliotecas
# ==============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
# 2. Criando o dataset
# ==============================
# Exemplo de frases em português com rótulos
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
        "Detestei, foi uma experiência ruim."
    ],
    "sentimento": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# ==============================
# 3. Separando treino e teste
# ==============================
X = df['frase']
y = df['sentimento']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4. Transformando textos em números com TF-IDF
# ==============================
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ==============================
# 5. Treinando o modelo
# ==============================
modelo = LogisticRegression()
modelo.fit(X_train_tfidf, y_train)

# ==============================
# 6. Avaliando o modelo
# ==============================
y_pred = modelo.predict(X_test_tfidf)
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia:.2f}")

# ==============================
# 7. Testando novas frases
# ==============================
novas_frases = [
    "O produto é excelente e chegou rápido!",
    "Estou decepcionado, serviço ruim."
]

novas_frases_tfidf = tfidf.transform(novas_frases)
previsoes = modelo.predict(novas_frases_tfidf)

for frase, pred in zip(novas_frases, previsoes):
    sentimento = "Positivo" if pred == 1 else "Negativo"
    print(f"Frase: '{frase}' -> Sentimento: {sentimento}")
