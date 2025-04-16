import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def unq_vals(col):
    # Accede a los valores únicos de la columna
    unique_vals = sorted(col.unique())
    
    # Solo procesar columnas que tengan al menos un valor único válido
    if len(unique_vals) > 0:
        if pd.api.types.is_numeric_dtype(col):
            unique_vals = [int(val) if isinstance(val, (np.integer, int)) else float(val) if isinstance(val, (np.floating, float)) else val for val in unique_vals]
        # Mostrar los valores únicos, pero solo los primeros 10 y últimos 10 si hay más de 20
        if len(unique_vals) <= 20:
            print(f"{col.name}: {unique_vals}")
        else:
            print(f"{col.name}: {unique_vals[:10] + ['.....'] + unique_vals[-10:]}")

def histogram_boxplot(data, xlabel = None, title = None, font_scale=2, figsize=(5,4), bins = None):
    
    # Definir tamaño letra
    sns.set(font_scale=font_scale)
    # Crear ventana para los subgráficos
    f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    # Crear boxplot
    sns.boxplot(x=data, ax=ax_box2)
    # Crear histograma
    sns.histplot(x=data, ax=ax_hist2, bins=bins) if bins else sns.histplot(x=data, ax=ax_hist2)
    # Pintar una línea con la media
    ax_hist2.axvline(np.mean(data),color='g',linestyle='-')
    # Pintar una línea con la mediana
    ax_hist2.axvline(np.median(data),color='y',linestyle='--')
    # Asignar título y nombre de eje si tal
    if xlabel: ax_hist2.set(xlabel=xlabel)
    if title: ax_box2.set(title=title, xlabel="")
    # Mostrar gráfico
    plt.show()

def cat_plot(col):
     if col.dtypes == 'category':
        #fig = px.bar(col.value_counts())
        fig = sns.countplot(x=col)
        return(fig)

def plot(col):
     if col.dtypes != 'category':
        histogram_boxplot(col, xlabel = col.name, title = 'Distibución continua')
     else:
        plt.clf()
        #cat_plot(col).show()
        cat_plot(col)

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
    """
    s : pd.Series
        Series to winsorize
    limits : tuple of float
        Tuple of the percentages to cut on each side of the array, 
        with respect to the number of unmasked data, as floats between 0. and 1
    """
    return s.clip(lower=s.quantile(limits[0], interpolation='lower'), 
                  upper=s.quantile(1-limits[1], interpolation='higher'))

from sklearn.ensemble import RandomForestClassifier

# Función que aplica Random Forest a los NAs
def imputer_RF(df, columna):
    # Verificar si hay valores faltantes en la columna
    if df[columna].isnull().sum() > 0:
        # Separar las filas con valores no nulos
        df_train = df.dropna(subset=[columna])
        
        # Variables predictoras y objetivo
        X_train = df_train.drop(columns=[columna])
        y_train = df_train[columna]
        
        # Convertir variables categóricas en numéricas usando one-hot encoding
        X_train = pd.get_dummies(X_train)
        
        # Entrenar un modelo
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Identificar filas con valores faltantes en la columna
        df_missing = df[df[columna].isnull()]
        X_missing = df_missing.drop(columns=[columna])
        
        # Asegurarse de que X_missing tenga las mismas columnas que X_train
        X_missing = pd.get_dummies(X_missing)
        X_missing = X_missing.reindex(columns=X_train.columns, fill_value=0)
        
        # Predecir y rellenar los valores faltantes
        df.loc[df[columna].isnull(), columna] = model.predict(X_missing)
    
    return df

from scipy.stats import boxcox

def boxcox_transf(df, col):
    min_val = df[col].min()
    
    # Asegurarse de que no haya valores negativos o cero en la columna antes de aplicar Box-Cox
    if min_val <= 0:
        df[col] += abs(min_val) + 1  # Para evitar valores <= 0
    
    # Aplicar la transformación Box-Cox
    y, lambda_ = boxcox(df[col])
    
    # Asignar la columna transformada a la original
    df[col] = y

    return lambda_

def transf_objpearson(df, columnas, objetivo):
    # Iteramos sobre las columnas que queremos transformar
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
        transf_cleaned = transf.dropna(subset=transf.columns)  # Eliminar filas con NaN en las transformaciones
        df_cleaned = df.loc[transf_cleaned.index, objetivo]  # Filtrar el objetivo para las filas limpias

        # Calculamos la correlación de Pearson entre cada transformación y el objetivo
        pearson_corr = {}
        for transf_col in transf_cleaned.columns:
            pearson_corr[transf_col] = np.corrcoef(transf_cleaned[transf_col].values, df_cleaned.values)[0, 1]

        # Convertimos el diccionario en DataFrame para poder graficar con Plotly
        pearson_df = pd.DataFrame(list(pearson_corr.items()), columns=["Transformación", "Coeficiente de Pearson"])

        # Graficamos con plotly.express para esta columna, con orientación horizontal
        fig = px.bar(pearson_df, x="Coeficiente de Pearson", y="Transformación", 
                     title=f"Correlación de Pearson de transformaciones de {col} respecto a {objetivo}", 
                     orientation='h')  # Gráfico horizontal
        fig.update_layout(xaxis_title="Coeficiente de Pearson", yaxis_title="Transformación")
        fig.show()

