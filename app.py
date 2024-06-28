import pandas as pd
import streamlit as st

# 1. Generar un DataFrame con los datos del fichero
df = pd.read_csv('titanic.csv')

st.title('Análisis de Datos del Titanic')

# 2. Mostrar información básica del DataFrame
st.header('Información básica del DataFrame')
st.write("Dimensiones del DataFrame:", df.shape)
st.write("Número de datos:", df.size)
st.write("Nombres de las columnas:", df.columns.tolist())
st.write("Nombres de las filas:", df.index.tolist())
st.write("Tipos de datos de las columnas:\n", df.dtypes)
st.write("Las 10 primeras filas:\n", df.head(10))
st.write("Las 10 últimas filas:\n", df.tail(10))
st.write("")

# 3. Mostrar por pantalla los datos del pasajero con identificador 148
st.header('Datos del pasajero con identificador 148')
st.write(df[df['PassengerId'] == 148])
st.write("")

# 4. Mostrar por pantalla las filas pares del DataFrame
st.header('Filas pares del DataFrame')
st.write(df.iloc[::2])
st.write("")

# 5. Mostrar por pantalla los nombres de las personas que iban en primera clase ordenadas alfabéticamente
st.header('Nombres de las personas en primera clase ordenadas alfabéticamente')
st.write(df[df['Pclass'] == 1]['Name'].sort_values())
st.write("")

# 6. Mostrar por pantalla el porcentaje de personas que sobrevivieron y murieron
st.header('Porcentaje de personas que sobrevivieron y murieron')
survival_counts = df['Survived'].value_counts(normalize=True) * 100
st.write(survival_counts)
st.write("")

# 7. Mostrar por pantalla el porcentaje de personas que sobrevivieron en cada clase
st.header('Porcentaje de personas que sobrevivieron en cada clase')
survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
st.write(survival_by_class)
st.write("")

# 8. Eliminar del DataFrame los pasajeros con edad desconocida
df_cleaned = df.dropna(subset=['Age']).copy()

# 9. Mostrar por pantalla la edad media de las mujeres que viajaban en cada clase
st.header('Edad media de las mujeres que viajaban en cada clase')
mean_age_women_by_class = df_cleaned[df_cleaned['Sex'] == 'female'].groupby('Pclass')['Age'].mean()
st.write(mean_age_women_by_class)
st.write("")

# 10. Añadir una nueva columna booleana para ver si el pasajero era menor de edad o no
df_cleaned.loc[:, 'Menor'] = df_cleaned['Age'] < 18

# 11. Mostrar por pantalla el porcentaje de menores y mayores de edad que sobrevivieron en cada clase
st.header('Porcentaje de menores y mayores de edad que sobrevivieron en cada clase')
survival_by_age_class = df_cleaned.groupby(['Pclass', 'Menor'])['Survived'].mean() * 100
st.write(survival_by_age_class)
st.write("")
