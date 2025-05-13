import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def num_graph(df):
    for i in df:
        fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.15, 0.85)}, figsize=(8, 5))   

        sns.boxplot(x=df[i], ax=ax_box, color='blue')
        ax_box.set(title=f"Distribución de '{df[i].name}'", xlabel='')

        sns.histplot(df[i], ax=ax_hist, bins=50, kde=True, color='blue')
        ax_hist.axvline(np.mean(df[i]), color='red', linestyle='-')
        ax_hist.axvline(np.median(df[i]), color='orange', linestyle='--')

        plt.show()

def cat_graph(df):
    for i in df:
        if df[i].nunique() > 20:
            print(f"La variable {df[i].name} tiene alta cardinalidad. Se decide no representarla.")
        else: 
            mode_value = df[i].mode()[0]

            plt.figure(figsize=(8, 5))

            sns.countplot(x=df[i], color='blue')
            plt.title(f"Gráfico de Barras '{df[i].name}'")
            plt.xticks(rotation=90)
            plt.show()


from scipy import stats

def manage_outliers(col,clas = 'check'):

     # Condición de asimetría y aplicación de criterio 1 según el caso
     if abs(col.skew()) < 1:
         criterio1 = abs((col - col.mean()) / col.std()) > 3
     else:
         criterio1 = abs((col - col.median()) / stats.median_abs_deviation(col)) > 6

     # Calcular primer cuartil     
     q1 = col.quantile(0.25)  
     # Calcular tercer cuartil  
     q3 = col.quantile(0.75)
     # Calculo de IQR
     IQR = q3 - q1

     # Calcular criterio 2 (general para cualquier asimetría)
     criterio2 = (col < (q1 - 3 * IQR)) | (col > (q3 + 3 * IQR))
     lower = col[criterio1 & criterio2 & (col < q1)].count() / col.dropna().count()
     upper = col[criterio1 & criterio2 & (col > q3)].count() / col.dropna().count()

     # Salida según el tipo deseado
     if clas == 'check':  # Para checkear los outliers en porcentajes
         return (lower * 100, upper * 100, (lower + upper) * 100)
     elif clas == 'winsor':  # Para winsorizar los outliers checkeados
         return winsorize_pd(col, (lower, upper))
     elif clas == 'miss':  # Para poner como missing los outliers checkeados
         col.loc[criterio1 & criterio2] = np.nan
         return col

def winsorize_pd(s, limits):

    return s.clip(lower=s.quantile(limits[0], interpolation='lower'), 
                  upper=s.quantile(1-limits[1], interpolation='higher'))

from scipy.stats import boxcox

def boxcox_transf(df, col):

    min_val = df[col].min()
    
    # Asegurarse de que no haya valores negativos o cero en la columna antes de aplicar Box-Cox
    if min_val <= 0:
        df[col] += abs(min_val) + 1
    
    y, lambda_ = boxcox(df[col])
    
    df[col] = y

def transf_pearson(df, columnas, objetivo):

    for col in columnas:
        vv = df[col].copy()

        # Calcular Box-Cox solo si los valores son positivos
        min_val = vv.min()
        if min_val <= 0:
            vv += abs(min_val) + 1

        y, lambda_ = boxcox(vv)

        # Crear un DataFrame con las transformaciones
        transf = pd.DataFrame({
            col + '_ident': vv,
            col + '_log': np.log(vv + 1),  # Evitar log(0)
            col + '_exp': np.exp(vv),
            col + '_sqrt': np.sqrt(vv),
            col + '_sqr': np.square(vv),
            col + '_cuarta': vv ** 4,
            col + '_raiz4': vv ** (1/4),
            col + '_boxcox': (np.log(vv) if lambda_ == 0 else (vv ** lambda_ - 1) / lambda_)
        })

        # Limpiar valores NaN de las transformaciones y objetivo para el cálculo de correlación
        transf_cleaned = transf.dropna(subset=transf.columns) 
        df_cleaned = df.loc[transf_cleaned.index, objetivo] 

        # Calculamos la correlación de Pearson entre cada transformación y el objetivo
        pearson_corr = {}
        for transf_col in transf_cleaned.columns:
            pearson_corr[transf_col] = np.corrcoef(transf_cleaned[transf_col].values, df_cleaned.values)[0, 1]

        # Convertimos el diccionario en DataFrame y ordenamos por coeficiente de Pearson (descendente)
        pearson_df = pd.DataFrame(list(pearson_corr.items()), columns=["Transformación", "Coeficiente de Pearson"])
        pearson_df = pearson_df.sort_values(by="Coeficiente de Pearson", ascending=False)

        # Crear el gráfico de barras horizontal con seaborn
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=pearson_df,
            x="Coeficiente de Pearson",
            y="Transformación",
            palette="viridis"
        )
        plt.title(f"Correlación de Pearson de transformaciones de {col} respecto a {objetivo}")
        plt.xlabel("Coeficiente de Pearson")
        plt.ylabel("Transformación")
        plt.tight_layout()

        plt.show()

