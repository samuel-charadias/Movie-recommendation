from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Carregar dados do filme (certifique-se de que seu DataFrame 'filmes' esteja carregado corretamente)
filmes = pd.read_csv('movies1.csv')# Exemplo de carregamento de dados
filmes['genres'] = filmes['genres'].fillna('')

# Criar a matriz TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(filmes['genres'])

# Calcular a similaridade do cosseno
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Criar uma série de índices de filmes para títulos
indices = pd.Series(filmes.index, index=filmes['name']).drop_duplicates()

# Função para obter recomendações por título
def get_recommendations(name, cosine_sim=cosine_sim):
    if name not in indices:
        raise KeyError(f"'{name}' não encontrado no banco de dados.")
    
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return filmes['name'].iloc[movie_indices]

# Função para obter recomendações por gênero
def get_recommendations_by_genre(genres):
    filtered_movies = filmes[filmes['genres'].str.contains(genres, case=False, na=False)]
    if filtered_movies.empty:
        raise KeyError("Gênero não encontrado")
    return filtered_movies['name'].head(10)

# Inicializar a aplicação Flask
app = Flask(__name__)

# Página inicial com opções de recomendação
@app.route('/')
def index():
    return render_template('index.html')

# Rota para recomendações por título
@app.route('/recommend_by_title', methods=['POST'])
def recommend_by_title():
    try:
        name = request.form['name']
        recommendations = get_recommendations(name)
        return render_template('recommendations.html', name=name, recommendations=recommendations)
    except KeyError as e:
        return render_template('error.html', message=str(e))
    except Exception as e:
        return render_template('error.html', message="Ocorreu um erro inesperado.")

# Rota para recomendações por gênero
@app.route('/recommend_by_genre', methods=['POST'])
def recommend_by_genre():
    try:
        genres = request.form['genres']
        recommendations = get_recommendations_by_genre(genres)
        return render_template('recommendations.html', genres=genres, recommendations=recommendations)
    except KeyError as e:
        return render_template('error.html', message=str(e))
    except Exception as e:
        return render_template('error.html', message="Ocorreu um erro inesperado.")

# Executar o servidor Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

