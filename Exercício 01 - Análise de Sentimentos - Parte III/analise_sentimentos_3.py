# ==============================
# 1. Importando bibliotecas
# ==============================
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ==============================
# 2. Garantindo que os recursos do NLTK estão disponíveis
# ==============================
try:
    stop_words = set(stopwords.words('portuguese'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('portuguese'))

try:
    stemmer = RSLPStemmer()
except LookupError:
    nltk.download('rslp')
    stemmer = RSLPStemmer()

# ==============================
# 3. Função de pré-processamento
# ==============================
def preprocess(texto):
    # 1. Converter para minúsculas
    texto = texto.lower()
    # 2. Remover pontuação
    texto = re.sub(r'[^\w\s]', '', texto)
    # 3. Remover stopwords e aplicar stemming
    palavras = texto.split()
    palavras_processadas = [stemmer.stem(palavra) for palavra in palavras if palavra not in stop_words]
    return ' '.join(palavras_processadas)

# ==============================
# 4. Criando o dataset
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
        # Frases extras
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
# 5. Aplicando pré-processamento
# ==============================
df['frase_limpa'] = df['frase'].apply(preprocess)

# ==============================
# 6. Separando treino e teste
# ==============================
X = df['frase_limpa']
y = df['sentimento']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 7. Vetorização TF-IDF
# ==============================
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ==============================
# 8. Treinando modelos
# ==============================
regressao_logistica = LogisticRegression()
regressao_logistica.fit(X_train_tfidf, y_train)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)

# ==============================
# 9. Avaliando os modelos
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
# 10. Função para testar novas frases
# ==============================
def testar_frases(modelo, frases):
    frases_limpa = [preprocess(frase) for frase in frases]
    frases_tfidf = tfidf.transform(frases_limpa)
    previsoes = modelo.predict(frases_tfidf)
    for frase, pred in zip(frases, previsoes):
        sentimento = "Positivo" if pred == 1 else "Negativo"
        print(f"Frase: '{frase}' -> Sentimento: {sentimento}")

# ==============================
# 11. Testando
# ==============================
novas_frases = [
    "O produto é excelente e chegou rápido!",
    "Estou decepcionado, serviço ruim."
]

print("=== Testando Regressão Logística com pré-processamento ===")
testar_frases(regressao_logistica, novas_frases)

print("\n=== Testando Naive Bayes com pré-processamento ===")
testar_frases(naive_bayes, novas_frases)
