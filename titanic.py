import pandas as pd

# 1. Generar un DataFrame con los datos del fichero
df = pd.read_csv('titanic.csv')

# 2. Mostrar por pantalla las dimensiones del DataFrame, el número de datos que contiene,
# los nombres de sus columnas y filas, los tipos de datos de las columnas, las 10 primeras filas y las 10 últimas filas
print("Dimensiones del DataFrame:", df.shape)
print("Número de datos:", df.size)
print("Nombres de las columnas:", df.columns)
print("Nombres de las filas:", df.index)
print("Tipos de datos de las columnas:\n", df.dtypes)
print("Las 10 primeras filas:\n", df.head(10))
print("Las 10 últimas filas:\n", df.tail(10))
print()

# 3. Mostrar por pantalla los datos del pasajero con identificador 148
print("Datos del pasajero con identificador 148:\n", df[df['PassengerId'] == 148])
print()

# 4. Mostrar por pantalla las filas pares del DataFrame
print("Filas pares del DataFrame:\n", df.iloc[::2])
print()

# 5. Mostrar por pantalla los nombres de las personas que iban en primera clase ordenadas alfabéticamente
print("Nombres de las personas que iban en primera clase ordenadas alfabéticamente:\n", df[df['Pclass'] == 1]['Name'].sort_values())
print()

# 6. Mostrar por pantalla el porcentaje de personas que sobrevivieron y murieron
survival_counts = df['Survived'].value_counts(normalize=True) * 100
print("Porcentaje de personas que sobrevivieron y murieron:\n", survival_counts)
print()

# 7. Mostrar por pantalla el porcentaje de personas que sobrevivieron en cada clase
survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
print("Porcentaje de personas que sobrevivieron en cada clase:\n", survival_by_class)
print()

# 8. Eliminar del DataFrame los pasajeros con edad desconocida
df_cleaned = df.dropna(subset=['Age']).copy()

# 9. Mostrar por pantalla la edad media de las mujeres que viajaban en cada clase
mean_age_women_by_class = df_cleaned[df_cleaned['Sex'] == 'female'].groupby('Pclass')['Age'].mean()
print("Edad media de las mujeres que viajaban en cada clase:\n", mean_age_women_by_class)
print()

# 10. Añadir una nueva columna booleana para ver si el pasajero era menor de edad o no
df_cleaned.loc[:, 'Menor'] = df_cleaned['Age'] < 18

# 11. Mostrar por pantalla el porcentaje de menores y mayores de edad que sobrevivieron en cada clase
survival_by_age_class = df_cleaned.groupby(['Pclass', 'Menor'])['Survived'].mean() * 100
print("Porcentaje de menores y mayores de edad que sobrevivieron en cada clase:\n", survival_by_age_class)
print()
