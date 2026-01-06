# ==============================
# 1. Importando bibliotecas
# ==============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ==============================
# 2. Criando o dataset
# ==============================
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
        # Frases extras para desafio
        "O livro é excelente e inspirador!",
        "Horrível, nunca mais compro neste lugar.",
        "A equipe foi muito atenciosa e prestativa.",
        "Péssimo atendimento, fiquei decepcionado.",
        "Maravilhoso, superou minhas expectativas!"
    ],
    "sentimento": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
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
# 5. Treinando modelos
# ==============================
# Regressão Logística
regressao_logistica = LogisticRegression()
regressao_logistica.fit(X_train_tfidf, y_train)

# Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)

# ==============================
# 6. Avaliando os modelos
# ==============================
for nome_modelo, modelo in [("Regressão Logística", regressao_logistica),
                            ("Naive Bayes", naive_bayes)]:
    y_pred = modelo.predict(X_test_tfidf)
    acuracia = accuracy_score(y_test, y_pred)
    print(f"{nome_modelo} - Acurácia: {acuracia:.2f}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Negativo", "Positivo"])
    disp.plot()
    print("\n")

# ==============================
# 7. Função para testar novas frases
# ==============================
def testar_frases(modelo, frases):
    frases_tfidf = tfidf.transform(frases)
    previsoes = modelo.predict(frases_tfidf)
    for frase, pred in zip(frases, previsoes):
        sentimento = "Positivo" if pred == 1 else "Negativo"
        print(f"Frase: '{frase}' -> Sentimento: {sentimento}")

# Testando a função com o usuário
novas_frases = [
    "O produto é excelente e chegou rápido!",
    "Estou decepcionado, serviço ruim."
]

print("=== Testando Regressão Logística ===")
testar_frases(regressao_logistica, novas_frases)

print("\n=== Testando Naive Bayes ===")
testar_frases(naive_bayes, novas_frases)
