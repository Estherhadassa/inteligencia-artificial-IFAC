# ==============================
# IMPORTS
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
# GARANTINDO RECURSOS NLTK
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
# FUNÇÃO DE PRÉ-PROCESSAMENTO
# ==============================
def preprocess(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    palavras = texto.split()
    palavras_processadas = [stemmer.stem(palavra) for palavra in palavras if palavra not in stop_words]
    return ' '.join(palavras_processadas)

# ==============================
# DATASET
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
        "O livro é excelente e inspirador!",
        "Horrível, nunca mais compro neste lugar.",
        "A equipe foi muito atenciosa e prestativa.",
        "Péssimo atendimento, fiquei decepcionado.",
        "Maravilhoso, superou minhas expectativas!"
    ],
    "sentimento": [1,1,0,0,1,0,1,0,1,0,1,0,1,0,1]
}

df = pd.DataFrame(data)
df['frase_limpa'] = df['frase'].apply(preprocess)

X = df['frase_limpa']
y = df['sentimento']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ==============================
# TREINAMENTO DOS MODELOS
# ==============================
regressao_logistica = LogisticRegression()
regressao_logistica.fit(X_train_tfidf, y_train)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)

# ==============================
# FUNÇÃO DE MATRIZ DE CONFUSÃO E ANÁLISE DE ERROS
# ==============================
def analisar_erros(modelo, X_test, y_test, nomes_classes=["Negativo","Positivo"]):
    y_pred = modelo.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Matriz de Confusão:")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
    disp.plot()
    
    # Identificando erros
    df_erro = pd.DataFrame({'frase': X_test, 'verdadeiro': y_test, 'predito': y_pred})
    falsos_positivos = df_erro[(df_erro['verdadeiro']==0) & (df_erro['predito']==1)]
    falsos_negativos = df_erro[(df_erro['verdadeiro']==1) & (df_erro['predito']==0)]
    
    print("\nFalsos Positivos (classificados como positivos mas são negativos):")
    print(falsos_positivos[['frase','verdadeiro','predito']], "\n")
    
    print("Falsos Negativos (classificados como negativos mas são positivos):")
    print(falsos_negativos[['frase','verdadeiro','predito']], "\n")
    
    return y_pred

# ==============================
# FUNÇÃO DE TESTE COM PROBABILIDADE E PESOS DAS PALAVRAS
# ==============================
def prever_sentimento(modelo, frases, tfidf, top_n=5):
    frases_limpa = [preprocess(f) for f in frases]
    X_tfidf = tfidf.transform(frases_limpa)
    
    probs = modelo.predict_proba(X_tfidf)
    preds = modelo.predict(X_tfidf)
    
    print("\n=== Resultados ===")
    for i, frase in enumerate(frases):
        classe = "Positivo" if preds[i]==1 else "Negativo"
        confianca = probs[i][preds[i]] * 100
        print(f"Frase: '{frase}'")
        print(f"Sentimento Previsto: {classe} ({confianca:.1f}%)")
        
        # Pesos das palavras mais relevantes (apenas para Regressão Logística)
        if isinstance(modelo, LogisticRegression):
            pesos = modelo.coef_[0]
            vocab = tfidf.get_feature_names_out()
            indices = X_tfidf[i].nonzero()[1]
            palavras = [(vocab[idx], pesos[idx]) for idx in indices]
            palavras = sorted(palavras, key=lambda x: abs(x[1]), reverse=True)[:top_n]
            print("Palavras mais relevantes:", palavras)
        print("")

# ==============================
# EXECUÇÃO
# ==============================
print("=== Analisando erros da Regressão Logística ===")
y_pred_rl = analisar_erros(regressao_logistica, X_test, y_test)

print("=== Analisando erros do Naive Bayes ===")
y_pred_nb = analisar_erros(naive_bayes, X_test, y_test)

# Teste de novas frases com probabilidade e palavras importantes
novas_frases = [
    "O produto é excelente e chegou rápido!",
    "Estou decepcionado, serviço ruim.",
    "O filme não foi ruim, mas poderia ser melhor."
]

print("\n=== Testando Regressão Logística com probabilidade e pesos ===")
prever_sentimento(regressao_logistica, novas_frases, tfidf)

print("\n=== Testando Naive Bayes com probabilidade ===")
prever_sentimento(naive_bayes, novas_frases, tfidf)
