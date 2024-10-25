import pandas as pd

filmes = pd.read_csv(r'c:\Users\JCD\Documents\Trabalho_Algebra\movies1.csv')
print(filmes.columns)

# Testar carregamento dos dados
print(filmes.head())  # Para garantir que o DataFrame 'filmes' está carregado corretamente
