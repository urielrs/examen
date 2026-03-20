import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Marketing AI", layout="wide")

# -----------------------------
# ESTILO ROJO
# -----------------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0f0f0f;
        color: #ffffff;
    }

    h1, h2, h3 {
        color: #ff2b2b;
    }

    section[data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }

    .stButton > button {
        background-color: #ff2b2b;
        color: white;
        border-radius: 8px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #cc0000;
        color: white;
    }

    div[data-baseweb="select"] > div {
        background-color: #1a1a1a;
        color: white;
    }

    .stSlider > div {
        color: #ff2b2b;
    }

    .stDataFrame {
        background-color: #1a1a1a;
    }

    .stAlert {
        background-color: #330000;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Marketing AI")
st.write("Aplicación web para análisis de ventas y clustering de clientes.")

uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file, encoding="latin1")

if uploaded_file:
    df = load_data(uploaded_file)
else:
    try:
        df = pd.read_csv("sales_data_sample.csv", encoding="latin1")
        st.info("Usando dataset incluido.")
    except Exception:
        st.warning("Sube un archivo CSV para continuar.")
        st.stop()

st.subheader("Vista previa del dataset")
st.dataframe(df.head())

# -----------------------------
# PREPROCESAMIENTO BASE
# -----------------------------
processed_df = df.copy()

if "ORDERDATE" in processed_df.columns:
    processed_df["ORDERDATE"] = pd.to_datetime(processed_df["ORDERDATE"], errors="coerce")

if "ORDERDATE" in processed_df.columns:
    processed_df["MONTH"] = processed_df["ORDERDATE"].dt.month
    processed_df["YEAR"] = processed_df["ORDERDATE"].dt.year

cols_drop = [
    "ADDRESSLINE1", "ADDRESSLINE2", "POSTALCODE", "CITY",
    "TERRITORY", "PHONE", "STATE", "CONTACTFIRSTNAME",
    "CONTACTLASTNAME", "CUSTOMERNAME", "ORDERNUMBER"
]

processed_df.drop(
    columns=[c for c in cols_drop if c in processed_df.columns],
    inplace=True,
    errors="ignore"
)

if "STATUS" in processed_df.columns:
    processed_df.drop(columns=["STATUS"], inplace=True)

for col in ["COUNTRY", "PRODUCTLINE", "DEALSIZE"]:
    if col in processed_df.columns:
        dummies = pd.get_dummies(processed_df[col], prefix=col)
        processed_df = pd.concat([processed_df, dummies], axis=1)
        processed_df.drop(columns=[col], inplace=True)

if "PRODUCTCODE" in processed_df.columns:
    processed_df["PRODUCTCODE"] = pd.Categorical(processed_df["PRODUCTCODE"]).codes

if "ORDERDATE" in processed_df.columns:
    processed_df["ORDERDATE"] = processed_df["ORDERDATE"].map(
        lambda x: x.toordinal() if pd.notnull(x) else np.nan
    )

numeric_df = processed_df.select_dtypes(include="number").copy()
numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

if numeric_df.empty or numeric_df.shape[1] < 2:
    st.error("No hay suficientes columnas numéricas para el análisis.")
    st.stop()

st.subheader("Dataset procesado")
st.dataframe(numeric_df.head())

# -----------------------------
# ESCALADO GENERAL
# -----------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# -----------------------------
# MENÚ DE GRÁFICOS
# -----------------------------
st.sidebar.title("Menú de visualizaciones")

opcion = st.sidebar.selectbox(
    "Selecciona el gráfico que quieres mostrar",
    [
        "Distribución por país",
        "Distribución por estado",
        "Distribución por línea de producto",
        "Distribución por tamaño de oferta",
        "Ventas por mes",
        "Ventas por año",
        "Método del codo",
        "Clusters con PCA",
        "Matriz de correlación",
        "Boxplot de ventas",
        "Histograma de ventas"
    ]
)

# -----------------------------
# GRÁFICOS
# -----------------------------
if opcion == "Distribución por país":
    if "COUNTRY" in df.columns:
        country_counts = df["COUNTRY"].value_counts().reset_index()
        country_counts.columns = ["COUNTRY", "COUNT"]

        fig = px.bar(country_counts, x="COUNTRY", y="COUNT", color="COUNTRY", title="Distribución por país")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("La columna COUNTRY no existe en el archivo.")

elif opcion == "Distribución por estado":
    if "STATUS" in df.columns:
        status_counts = df["STATUS"].value_counts().reset_index()
        status_counts.columns = ["STATUS", "COUNT"]

        fig = px.bar(status_counts, x="STATUS", y="COUNT", color="STATUS", title="Distribución por estado")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("La columna STATUS no existe en el archivo.")

elif opcion == "Distribución por línea de producto":
    if "PRODUCTLINE" in df.columns:
        product_counts = df["PRODUCTLINE"].value_counts().reset_index()
        product_counts.columns = ["PRODUCTLINE", "COUNT"]

        fig = px.bar(product_counts, x="PRODUCTLINE", y="COUNT", color="PRODUCTLINE", title="Distribución por línea de producto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("La columna PRODUCTLINE no existe en el archivo.")

elif opcion == "Distribución por tamaño de oferta":
    if "DEALSIZE" in df.columns:
        dealsize_counts = df["DEALSIZE"].value_counts().reset_index()
        dealsize_counts.columns = ["DEALSIZE", "COUNT"]

        fig = px.pie(dealsize_counts, names="DEALSIZE", values="COUNT", title="Distribución por tamaño de oferta")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("La columna DEALSIZE no existe en el archivo.")

elif opcion == "Ventas por mes":
    if "ORDERDATE" in df.columns and "SALES" in df.columns:
        temp_df = df.copy()
        temp_df["ORDERDATE"] = pd.to_datetime(temp_df["ORDERDATE"], errors="coerce")
        temp_df["MONTH"] = temp_df["ORDERDATE"].dt.month

        monthly_sales = temp_df.groupby("MONTH", dropna=True)["SALES"].sum().reset_index()

        fig = px.line(monthly_sales, x="MONTH", y="SALES", markers=True, title="Ventas por mes")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Se necesitan las columnas ORDERDATE y SALES.")

elif opcion == "Ventas por año":
    if "ORDERDATE" in df.columns and "SALES" in df.columns:
        temp_df = df.copy()
        temp_df["ORDERDATE"] = pd.to_datetime(temp_df["ORDERDATE"], errors="coerce")
        temp_df["YEAR"] = temp_df["ORDERDATE"].dt.year

        yearly_sales = temp_df.groupby("YEAR", dropna=True)["SALES"].sum().reset_index()

        fig = px.bar(yearly_sales, x="YEAR", y="SALES", color="YEAR", title="Ventas por año")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Se necesitan las columnas ORDERDATE y SALES.")

elif opcion == "Método del codo":
    inertias = []
    max_k = min(10, len(numeric_df) - 1) if len(numeric_df) > 2 else 2
    max_k = max(max_k, 2)
    k_values = list(range(1, max_k + 1))

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)

    elbow_df = pd.DataFrame({"Clusters": k_values, "Inercia": inertias})

    fig = px.line(elbow_df, x="Clusters", y="Inercia", markers=True, title="Método del codo")
    st.plotly_chart(fig, use_container_width=True)

elif opcion == "Clusters con PCA":
    columnas_cluster = ["QUANTITYORDERED","PRICEEACH","SALES","MSRP","MONTH_ID","YEAR_ID"]
    columnas_existentes = [col for col in columnas_cluster if col in df.columns]

    if len(columnas_existentes) >= 2:
        cluster_df = df[columnas_existentes].copy().dropna()

        if len(cluster_df) < 3:
            st.warning("No hay suficientes registros válidos para generar clusters.")
        else:
            scaler_cluster = StandardScaler()
            scaled_cluster_data = scaler_cluster.fit_transform(cluster_df)

            max_clusters = max(2, min(8, len(cluster_df)))
            valor_default = 5 if max_clusters >= 5 else max_clusters

            n_clusters = st.slider("Número de clusters", 2, max_clusters, valor_default)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_cluster_data)

            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(scaled_cluster_data)

            pca_df = pd.DataFrame(pca_components, columns=["PCA1", "PCA2"])
            pca_df["cluster"] = clusters.astype(str)

            fig = px.scatter(pca_df, x="PCA1", y="PCA2", color="cluster",
                             title="Visualización de clusters con PCA",
                             labels={"cluster": "Cluster"})
            st.plotly_chart(fig, use_container_width=True)

            st.write("Variables usadas para clustering:", columnas_existentes)
    else:
        st.warning("No hay suficientes columnas numéricas adecuadas para clustering.")

elif opcion == "Matriz de correlación":
    columnas_corr = ["QUANTITYORDERED","PRICEEACH","SALES","MSRP","MONTH_ID","YEAR_ID"]
    columnas_existentes = [col for col in columnas_corr if col in df.columns]

    if len(columnas_existentes) >= 2:
        corr = df[columnas_existentes].corr(numeric_only=True)

        fig = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis",
                        title="Matriz de correlación")
        st.plotly_chart(fig, use_container_width=True)

        st.write("Variables usadas en la matriz:", columnas_existentes)
    else:
        st.warning("No hay suficientes columnas numéricas adecuadas.")

elif opcion == "Boxplot de ventas":
    if "SALES" in df.columns:
        fig = px.box(df, y="SALES", title="Boxplot de ventas")
        st.plotly_chart(fig, use_container_width=True)

elif opcion == "Histograma de ventas":
    if "SALES" in df.columns:
        fig = px.histogram(df, x="SALES", nbins=30, title="Histograma de ventas")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# DESCARGA
# -----------------------------
st.subheader("Descargar resultados")

columnas = ["QUANTITYORDERED","PRICEEACH","SALES","MSRP","MONTH_ID","YEAR_ID"]
columnas_existentes = [c for c in columnas if c in df.columns]

if len(columnas_existentes) >= 2:
    download_df = df[columnas_existentes].dropna()

    if len(download_df) >= 3:
        scaled = StandardScaler().fit_transform(download_df)
        kmeans = KMeans(n_clusters=5 if len(download_df)>=5 else 2, random_state=42, n_init=10)
        download_df["cluster"] = kmeans.fit_predict(scaled)

        csv = download_df.to_csv(index=False).encode("utf-8")

        st.download_button("Descargar CSV con clusters", csv, "clusters_marketing_ai.csv", "text/csv")