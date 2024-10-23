import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Passo 1: Gerar dados aleatórios refinados
np.random.seed(42)  # Para reprodutibilidade

# Criar um dataframe com 1000 registros simulados
n_pacientes = 10000
nomes = [f'Paciente_{i}' for i in range(1, n_pacientes + 1)]
idades = np.random.randint(5, 81, size=n_pacientes)  # Idades entre 5 e 80
generos = np.random.choice(['Masculino', 'Feminino'], size=n_pacientes)
glicemia_glicada = np.round(np.random.uniform(4.0, 17.0, size=n_pacientes), 2)  # Valores de 4.0 a 17.0
pressao_art = np.random.randint(90, 180, size=n_pacientes)
colesterol = np.random.randint(150, 300, size=n_pacientes)
historico_diabetes = np.random.choice(['Sim', 'Não'], size=n_pacientes)

# Novo: Histórico de outras doenças (comorbidades)
historico_doencas = np.random.choice(['Hipertensão', 'Doença Cardíaca', 'Nenhuma', 'Asma'], size=n_pacientes)

# Novo: Uso de medicamentos baseado no CID
medicamentos = {
    'E10': 'Insulina',
    'E11': 'Metformina',
    'I10': 'Betabloqueador',
    'I25': 'AAS',
    'E12': 'Saxagliptina',
    'F32': 'Antidepressivo',
    'K70': 'Antibiótico',
    'I10.9': 'Diurético',
    'E78': 'Estatina',
}

# Gerar dados aleatórios de CID
cids = ['E11', 'E10', 'E12', 'I10', 'I25', 'E78', 'F32', 'K70',
        'E10.9', 'E11.9', 'E12.9', 'I10.9', 'I20.9', 'I25.9']  # Exemplos de CID
df = pd.DataFrame({
    'Nome': nomes,
    'Idade': idades,
    'Genero': generos,
    'Glicemia': glicemia_glicada,
    'Pressao_Art': pressao_art,
    'Colesterol': colesterol,
    'Historico_Diabetes': historico_diabetes,
    'Historico_Doencas': historico_doencas,
    'CID': np.random.choice(cids, size=n_pacientes),
})

# Associar medicamentos ao CID
df['Medicamento'] = df['CID'].apply(lambda x: medicamentos.get(x, 'Nenhum'))

# Novo: Simulação de resultados anteriores de exames (últimos 3 anos)
df['Glicemia_Anterior'] = df['Glicemia'] - np.random.uniform(0, 2, size=n_pacientes)  # Simular variação
df['Pressao_Art_Anterior'] = df['Pressao_Art'] - np.random.randint(0, 10, size=n_pacientes)
df['Colesterol_Anterior'] = df['Colesterol'] - np.random.randint(0, 50, size=n_pacientes)

# Fatores de risco
df['Obesidade'] = np.random.choice(['Sim', 'Não'], size=n_pacientes, p=[0.3, 0.7])
df['Tabagismo'] = np.random.choice(['Sim', 'Não'], size=n_pacientes, p=[0.2, 0.8])
df['Sedentarismo'] = np.random.choice(['Sim', 'Não'], size=n_pacientes, p=[0.4, 0.6])
df['Historico_Familiar'] = np.random.choice(['Sim', 'Não'], size=n_pacientes, p=[0.5, 0.5])

# Codificar CIDS
label_encoder = LabelEncoder()
df['CID_codificado'] = label_encoder.fit_transform(df['CID'])

# Clusterização
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['CID_codificado']])

# Identificação de Outliers (glicemia fora do intervalo)
df['zscore'] = stats.zscore(df['Glicemia'])
outliers = df[df['zscore'].abs() > 2]

# Classificação do risco de diabetes com base em fatores adicionais
def classificar_risco(glicemia, historico_doencas, obesidade, tabagismo):
    risco = 0
    if glicemia > 6.4:
        risco += 1
    if historico_doencas != 'Nenhuma':
        risco += 1
    if obesidade == 'Sim' or tabagismo == 'Sim':
        risco += 1
    if risco >= 2:
        return 'Alto Risco'
    elif risco == 1:
        return 'Risco Moderado'
    else:
        return 'Sem Risco'

df['Risco Diabetes'] = df.apply(lambda row: classificar_risco(
    row['Glicemia'], row['Historico_Doencas'], row['Obesidade'], row['Tabagismo']), axis=1)

# Verificar os dados refinados
print(df.head())

# Passo 2: Visualizações e Análises

# Gráfico de barras comparando o risco por gênero
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Risco Diabetes', hue='Genero', palette='viridis')
plt.title('Distribuição do Risco de Diabetes por Gênero')
plt.xlabel('Risco de Diabetes')
plt.ylabel('Número de Pacientes')
plt.show()

# Gráfico de dispersão mostrando a relação entre idade e glicemia glicada com fatores de risco
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Idade', y='Glicemia', hue='Risco Diabetes', style='Obesidade', palette='coolwarm')
plt.title('Idade vs Glicemia com Risco de Diabetes e Fatores de Risco')
plt.xlabel('Idade')
plt.ylabel('Glicemia Glicada')
plt.show()

# Gráfico de densidade comparando glicemia glicada entre os gêneros
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Glicemia', hue='Genero', fill=True, palette='crest')
plt.title('Distribuição da Glicemia Glicada por Gênero')
plt.xlabel('Glicemia Glicada')
plt.ylabel('Densidade')
plt.show()