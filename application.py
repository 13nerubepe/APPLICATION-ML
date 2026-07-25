"""
Application de Prédiction d'Octroi de Prêt
--------------------------------------------
Application Streamlit professionnelle permettant :
- L'exploration des données (valeurs manquantes, statistiques, visualisations)
- Le prétraitement automatique des données
- L'entraînement et la comparaison de plusieurs modèles de Machine Learning
- La prédiction en temps réel de l'octroi d'un prêt pour un nouveau client
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)



import io

# ----------------------------------------------------------------------------
# CONFIGURATION GENERALE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prédiction de Prêt Bancaire",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CAT_COLS = ["Gender", "Married", "Dependents", "Education",
            "Self_Employed", "Property_Area"]
NUM_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History"]
TARGET = "Loan_Status"
ID_COL = "Loan_ID"


# ============================
# CHARGEMENT MODELE FINAL
# ============================

@st.cache_resource
def load_model():

    model = joblib.load("svm_smote_loan.pkl")

    encoder = joblib.load("encoder.pkl")

    scaler = joblib.load("scaler.pkl")

    imputer_num = joblib.load("imputer_num.pkl")

    imputer_cat = joblib.load("imputer_cat.pkl")

    return (
        model,
        encoder,
        scaler,
        imputer_num,
        imputer_cat
    )


model, encoder, scaler, imputer_num, imputer_cat = load_model()

# ----------------------------------------------------------------------------
# STYLE PERSONNALISE
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a3c6e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        color: #1a3c6e;
    }
    .stButton>button {
        background-color: #1a3c6e;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #244d8c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# FONCTIONS UTILITAIRES
# ----------------------------------------------------------------------------
@st.cache_data
def generate_sample_data(n=614, seed=42):
    """Génère un jeu de données synthétique reproduisant le schéma et les
    taux de valeurs manquantes du dataset 'Loan Prediction' classique."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n, p=[0.8, 0.2])
    married = rng.choice(["Yes", "No"], n, p=[0.65, 0.35])
    dependents = rng.choice(["0", "1", "2", "3+"], n, p=[0.58, 0.17, 0.17, 0.08])
    education = rng.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22])
    self_employed = rng.choice(["No", "Yes"], n, p=[0.86, 0.14])
    applicant_income = rng.gamma(shape=3.0, scale=1800, size=n).round(0)
    coapplicant_income = rng.choice(
        [0], n, p=[1.0]
    ).astype(float)
    mask_co = rng.random(n) < 0.55
    coapplicant_income[mask_co] = rng.gamma(2.0, 900, size=mask_co.sum()).round(0)
    loan_amount = (applicant_income / 25 + rng.normal(20, 40, n)).clip(9, 700).round(0)
    loan_term = rng.choice([360, 180, 120, 84, 300, 60, 36],
                           n, p=[0.83, 0.06, 0.03, 0.03, 0.02, 0.02, 0.01])
    credit_history = rng.choice([1.0, 0.0], n, p=[0.84, 0.16])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], n, p=[0.38, 0.38, 0.24])

    score = (
        (credit_history == 1.0).astype(float) * 3
        + (education == "Graduate").astype(float)
        + (applicant_income > 3000).astype(float)
        - (loan_amount > 200).astype(float)
        + rng.normal(0, 1.2, n)
    )
    loan_status = np.where(score > 2.0, "Y", "N")

    df = pd.DataFrame({
        "Loan_ID": [f"LP{100000+i}" for i in range(n)],
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term.astype(float),
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status,
    })

    # Injecter des valeurs manquantes selon les taux fournis par l'utilisateur
    missing_rates = {
        "Gender": 0.021173, "Married": 0.004886, "Dependents": 0.024430,
        "Self_Employed": 0.052117, "LoanAmount": 0.035831,
        "Loan_Amount_Term": 0.022801, "Credit_History": 0.081433,
    }
    for col, rate in missing_rates.items():
        idx = rng.choice(df.index, size=int(round(rate * n)), replace=False)
        df.loc[idx, col] = np.nan

    return df


def missing_values_table(df):
    total = df.isnull().sum()
    percent = (df.isnull().sum() / len(df) * 100)
    table = pd.DataFrame({
        "Colonne": total.index,
        "Valeurs manquantes": total.values,
        "Pourcentage (%)": percent.values.round(2)
    }).sort_values("Pourcentage (%)", ascending=False).reset_index(drop=True)
    return table


def preprocess(df, fit=True, encoders=None, imputers=None):
    """Nettoie, impute et encode le dataframe. Retourne X, y, encoders, imputers."""
    data = df.copy()

    if ID_COL in data.columns:
        data = data.drop(columns=[ID_COL])

    y = None
    if TARGET in data.columns:
        y = data[TARGET].map({"Y": 1, "N": 0, 1: 1, 0: 0})
        data = data.drop(columns=[TARGET])

    if fit:
        encoders = {}
        imputers = {"cat": SimpleImputer(strategy="most_frequent"),
                    "num": SimpleImputer(strategy="median")}

    present_cat = [c for c in CAT_COLS if c in data.columns]
    present_num = [c for c in NUM_COLS if c in data.columns]

    if fit:
        data[present_cat] = imputers["cat"].fit_transform(data[present_cat])
        data[present_num] = imputers["num"].fit_transform(data[present_num])
    else:
        data[present_cat] = imputers["cat"].transform(data[present_cat])
        data[present_num] = imputers["num"].transform(data[present_num])

    for col in present_cat:
        if fit:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            data[col] = data[col].astype(str).map(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            data[col] = le.transform(data[col])

    return data, y, encoders, imputers


def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Régression Logistique": LogisticRegression(max_iter=5000),
        "Arbre de Décision": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Forêt Aléatoire": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "y_pred": preds,
            "y_proba": proba,
        }
    return results


# ----------------------------------------------------------------------------
# ETAT DE SESSION
# ----------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None
if "encoders" not in st.session_state:
    st.session_state.encoders = None
if "imputers" not in st.session_state:
    st.session_state.imputers = None
if "best_model_name" not in st.session_state:
    st.session_state.best_model_name = None

if "historique_predictions" not in st.session_state:
    st.session_state.historique_predictions = pd.DataFrame(
        columns=[
            "Date",
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
            "Decision",
            "Probabilité approbation",
            "Confiance"
        ]
    )

# ----------------------------------------------------------------------------
# BARRE LATERALE - NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 🏦 PREDICTION BANCAIRE")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📊 Exploration des données", "🧠 Modélisation","📋 Historique des prédictions", "🔮 Prédiction"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Données")
uploaded_file = st.sidebar.file_uploader("Charger un fichier CSV", type=["csv"])

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
elif st.session_state.df is None:
    st.session_state.df = generate_sample_data()
    # st.sidebar.info("Jeu de données de démonstration chargé.")

# if st.sidebar.button("🔄 Régénérer les données de démo"):
#     st.session_state.df = generate_sample_data(seed=np.random.randint(0, 10000))
#     st.session_state.results = None

# st.sidebar.markdown("---")
# st.sidebar.caption("Développé avec Streamlit • Machine Learning")

df = st.session_state.df

# ----------------------------------------------------------------------------
# PAGE 1 - ACCUEIL
# ----------------------------------------------------------------------------

if page == "🏠 Accueil":

    # ==============================
    # HERO SECTION
    # ==============================

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg,#1a3c6e,#2563eb);
            padding:40px;
            border-radius:20px;
            color:white;
            text-align:center;
            margin-bottom:30px;
        ">

        <h1 style="font-size:45px;">
        🏦 Smart Loan AI
        </h1>

        <h3>
        Système intelligent de prédiction d'octroi de prêt bancaire
        </h3>

        <p style="font-size:18px;">
        Une solution basée sur le Machine Learning permettant
        d'analyser automatiquement les dossiers clients
        et d'estimer la probabilité d'approbation d'un crédit.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ==============================
    # STATISTIQUES PRINCIPALES
    # ==============================

    st.markdown("## 📊 Vue générale")


    col1, col2, col3, col4 = st.columns(4)


    with col1:
        st.metric(
            "📁 Dossiers analysés",
            len(df)
        )


    with col2:

        st.metric(
            "🔢 Variables utilisées",
            df.shape[1]
        )


    with col3:

        approved = (
            (df[TARGET]=="Y").sum()
            if TARGET in df.columns
            else 0
        )

        st.metric(
            "✅ Prêts approuvés",
            approved
        )


    with col4:

        rate = (
            round(
                approved / len(df)*100,
                1
            )
            if TARGET in df.columns and len(df)>0
            else 0
        )

        st.metric(
            "📈 Taux d'approbation",
            f"{rate}%"
        )



    st.divider()



    # ==============================
    # PRESENTATION APPLICATION
    # ==============================


    st.markdown(
        "## Une aide intelligente à la décision"
    )


    col1,col2 = st.columns(2)


    with col1:

        st.write(
        """
        Cette application utilise des algorithmes de Machine Learning
        afin d'aider les institutions financières dans l'évaluation
        automatique des demandes de crédit.

        Le système analyse plusieurs critères :

        - 💰 Revenus du demandeur
        - 🏠 Zone de résidence
        - 💳 Historique de crédit
        - 👨‍👩‍👧 Situation familiale
        - 🎓 Niveau d'éducation
        """
        )


    with col2:

        st.markdown(
        """
    <div style="
    background:#f8fafc;
    padding:25px;
    border-radius:15px;
    border:1px solid #e5e7eb;
    color:#1f2937;
    text-align:center;
    ">

    <img src="assets/banqueimage.jpg"
    width="180"
    style="
    border-radius:15px;
    margin-bottom:15px;
    ">

    <h3 style="color:#1a3c6e;">
     Intelligence artificielle
    </h3>

    <p style="color:#374151;">
    Le modèle analyse le profil du client et fournit :
    </p>

    <ul style="
    text-align:left;
    color:#374151;
    ">

    <li>Une décision : Approuvé / Refusé</li>

    <li> Une probabilité d'approbation</li>

    <li> Un niveau de confiance</li>

    </ul>

    </div>
    """,
    unsafe_allow_html=True
    )



    # ==============================
    # FONCTIONNALITES
    # ==============================


    st.markdown(
        "## ⚙️ Fonctionnalités"
    )


    f1,f2,f3 = st.columns(3)


    with f1:

        st.info(
        """
        ###  Exploration

        Analyse des données,
        valeurs manquantes,
        statistiques et visualisations.
        """
        )


    with f2:

        st.success(
        """
        ###  Machine Learning

        Entraînement,
        comparaison des modèles
        et évaluation des performances.
        """
        )


    with f3:

        st.warning(
        """
        ###  Prédiction

        Simulation instantanée
        d'une nouvelle demande de prêt.
        """
        )



    # ==============================
    # PROCESSUS
    # ==============================


    st.markdown(
        "## 🔄 Comment ça fonctionne ?"
    )


    steps = st.columns(4)


    with steps[0]:
        st.markdown(
        """
        ### 1️⃣ Données

        Importation du dataset bancaire
        """
        )


    with steps[1]:
        st.markdown(
        """
        ### 2️⃣ Prétraitement

        Nettoyage et transformation
        """
        )


    with steps[2]:
        st.markdown(
        """
        ### 3️⃣ Modèle IA

        Analyse automatique
        """
        )


    with steps[3]:
        st.markdown(
        """
        ### 4️⃣ Décision

        Résultat et probabilité
        """
        )



    st.divider()



    # ==============================
    # GRAPHIQUE
    # ==============================


    if TARGET in df.columns:

        st.subheader(
            " Répartition des décisions historiques"
        )


        fig = px.pie(
            df,
            names=TARGET,
            hole=0.5,
            color=TARGET,
            color_discrete_map={
                "Y":"#22c55e",
                "N":"#ef4444"
            }
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )



    # ==============================
    # APERCU DONNEES
    # ==============================


    with st.expander(
        " Aperçu des données"
    ):

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

# ----------------------------------------------------------------------------
# PAGE 2 - EXPLORATION DES DONNEES
# ----------------------------------------------------------------------------
elif page == "📊 Exploration des données":
    st.markdown('<p class="main-header">📊 Exploration des Données</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyse descriptive et qualité des données</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🧩 Valeurs manquantes", "📈 Statistiques", "🔍 Visualisations"])

    with tab1:
        st.subheader("Valeurs manquantes par colonne")
        mv_table = missing_values_table(df)
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.dataframe(mv_table, use_container_width=True, hide_index=True)
            total_missing = int(mv_table["Valeurs manquantes"].sum())
            st.metric("Total de cellules manquantes", total_missing)
        with c2:
            mv_plot = mv_table[mv_table["Valeurs manquantes"] > 0]
            if not mv_plot.empty:
                fig = px.bar(
                    mv_plot, x="Pourcentage (%)", y="Colonne", orientation="h",
                    text="Pourcentage (%)", color="Pourcentage (%)",
                    color_continuous_scale="Blues",
                    title="Taux de valeurs manquantes (%)"
                )
                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig.update_layout(yaxis={"categoryorder": "total ascending"},
                                   coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("Aucune valeur manquante détectée dans ce jeu de données.")

    with tab2:
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
        st.subheader("Types de données")
        dtypes_df = pd.DataFrame({
            "Colonne": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Valeurs uniques": [df[c].nunique() for c in df.columns]
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            num_col = st.selectbox("Variable numérique", NUM_COLS, key="num_viz")
            fig = px.histogram(df, x=num_col, color=TARGET if TARGET in df.columns else None,
                                marginal="box", barmode="overlay", opacity=0.7,
                                color_discrete_map={"Y": "#1a3c6e", "N": "#e15759"},
                                title=f"Distribution de {num_col}")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            cat_col = st.selectbox("Variable catégorielle", CAT_COLS, key="cat_viz")
            grp = df.groupby([cat_col, TARGET]).size().reset_index(name="count") \
                if TARGET in df.columns else df[cat_col].value_counts().reset_index()
            if TARGET in df.columns:
                fig2 = px.bar(grp, x=cat_col, y="count", color=TARGET, barmode="group",
                              color_discrete_map={"Y": "#1a3c6e", "N": "#e15759"},
                              title=f"{cat_col} vs statut du prêt")
            else:
                fig2 = px.bar(grp, x="index", y=cat_col, title=f"Répartition de {cat_col}")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Matrice de corrélation (variables numériques)")
        num_present = [c for c in NUM_COLS if c in df.columns]
        corr = df[num_present].corr()
        fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, title="Corrélations")
        st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 3 - MODELISATION
# ----------------------------------------------------------------------------
elif page == " Modélisation":

    st.markdown(
        '<p class="main-header"> Laboratoire de Modélisation IA</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
        background:#f8fafc;
        padding:20px;
        border-radius:15px;
        border:1px solid #e5e7eb;
        margin-bottom:20px;
        ">

        <h3> Entraînement des modèles Machine Learning</h3>

        <p>
        Cette section permet de préparer les données, entraîner plusieurs
        algorithmes de classification et sélectionner le modèle le plus performant
        pour prédire l'octroi d'un prêt bancaire.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    if TARGET not in df.columns:

        st.error(
            " La colonne cible Loan_Status est absente."
        )

    else:


        # ============================
        # PARAMETRES
        # ============================

        st.subheader(" Configuration de l'entraînement")


        col1, col2, col3 = st.columns(3)


        with col1:

            test_size = st.slider(
                "Taille du jeu de test",
                0.1,
                0.4,
                0.2,
                0.05
            )


        with col2:

            random_state = st.number_input(
                "Random State",
                value=42
            )


        with col3:

            st.info(
                """
                 Objectif :

                Trouver le modèle avec le meilleur F1-score.
                """
            )



        st.divider()



        # ============================
        # ENTRAINEMENT
        # ============================


        if st.button(
            " Lancer l'entraînement",
            use_container_width=True
        ):


            progress = st.progress(0)


            with st.spinner(
                "Préparation des données et entraînement..."
            ):


                progress.progress(20)


                X, y, encoders, imputers = preprocess(
                    df,
                    fit=True
                )


                progress.progress(40)



                X_train, X_test, y_train, y_test = train_test_split(

                    X,
                    y,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=y

                )


                progress.progress(60)



                results = train_models(
                    X_train,
                    X_test,
                    y_train,
                    y_test
                )


                progress.progress(90)



                st.session_state.results = results

                st.session_state.encoders = encoders

                st.session_state.imputers = imputers

                st.session_state.y_test = y_test



                best = max(
                    results,
                    key=lambda k: results[k]["f1"]
                )


                st.session_state.best_model_name = best


                progress.progress(100)



            st.success(
                f"✅ Entraînement terminé ! Meilleur modèle : {best}"
            )



        # ============================
        # RESULTATS
        # ============================


        if st.session_state.results:


            results = st.session_state.results



            best_model = st.session_state.best_model_name



            best_score = results[best_model]["f1"]



            # KPI CARDS

            col1,col2,col3,col4 = st.columns(4)


            with col1:

                st.metric(
                    " Modèles testés",
                    len(results)
                )


            with col2:

                st.metric(
                    "🏆 Meilleur modèle",
                    best_model
                )


            with col3:

                st.metric(
                    "⭐ F1-score",
                    f"{best_score:.3f}"
                )


            with col4:

                st.metric(
                    "📂 Variables",
                    X.shape[1]
                )



            st.divider()



            tab1,tab2,tab3,tab4 = st.tabs(
                [
                    " Performances",
                    " Matrice confusion",
                    " Courbe ROC",
                    " Importance variables"
                ]
            )



            # -------------------------
            # PERFORMANCE
            # -------------------------

            with tab1:


                comp_df = pd.DataFrame({

                    name:{

                        "Accuracy":r["accuracy"],

                        "Precision":r["precision"],

                        "Recall":r["recall"],

                        "F1-score":r["f1"]

                    }

                    for name,r in results.items()

                }).transpose().round(3)



                st.dataframe(

                    comp_df.style
                    .highlight_max(
                        axis=0,
                        color="#bbf7d0"
                    ),

                    use_container_width=True

                )



                fig = px.bar(

                    comp_df.reset_index(),

                    x="index",

                    y=[
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1-score"
                    ],

                    barmode="group",

                    title="Comparaison des modèles"

                )


                st.plotly_chart(
                    fig,
                    use_container_width=True
                )



            # -------------------------
            # MATRICE CONFUSION
            # -------------------------

            with tab2:


                model_choice = st.selectbox(

                    "Choisir un modèle",

                    list(results.keys())

                )


                cm = confusion_matrix(

                    st.session_state.y_test,

                    results[model_choice]["y_pred"]

                )



                fig_cm = px.imshow(

                    cm,

                    text_auto=True,

                    color_continuous_scale="Blues",

                    labels={

                        "x":"Prédit",

                        "y":"Réel"

                    }

                )


                st.plotly_chart(

                    fig_cm,

                    use_container_width=True

                )




            # -------------------------
            # ROC
            # -------------------------

            with tab3:


                model_choice = st.selectbox(

                    "Modèle ROC",

                    list(results.keys()),

                    key="roc"

                )


                fpr,tpr,_ = roc_curve(

                    st.session_state.y_test,

                    results[model_choice]["y_proba"]

                )


                roc_auc = auc(fpr,tpr)



                fig = go.Figure()



                fig.add_trace(

                    go.Scatter(

                        x=fpr,

                        y=tpr,

                        mode="lines",

                        name=f"AUC={roc_auc:.3f}"

                    )

                )



                fig.add_trace(

                    go.Scatter(

                        x=[0,1],

                        y=[0,1],

                        mode="lines",

                        line=dict(
                            dash="dash"
                        )

                    )

                )



                fig.update_layout(

                    title="Courbe ROC",

                    xaxis_title="Faux positifs",

                    yaxis_title="Vrais positifs"

                )



                st.plotly_chart(

                    fig,

                    use_container_width=True

                )




            # -------------------------
            # IMPORTANCE
            # -------------------------

            with tab4:


                model_choice = st.selectbox(

                    "Modèle",

                    list(results.keys()),

                    key="importance"

                )


                selected = results[model_choice]["model"]



                if hasattr(
                    selected,
                    "feature_importances_"
                ):


                    imp = pd.DataFrame({

                        "Variable":X.columns,

                        "Importance":
                        selected.feature_importances_

                    }).sort_values(

                        "Importance",

                        ascending=True

                    )



                    fig = px.bar(

                        imp,

                        x="Importance",

                        y="Variable",

                        orientation="h",

                        title="Importance des variables"

                    )



                    st.plotly_chart(

                        fig,

                        use_container_width=True

                    )


                else:

                    st.warning(
                        "Ce modèle ne fournit pas d'importance des variables."
                    )



        else:

            st.info(
                " Cliquez sur 'Lancer l'entraînement' pour commencer."
            )

# ----------------------------------------------------------------------------
# PAGE 4 - PREDICTION
# ----------------------------------------------------------------------------

elif page == "🔮 Prédiction":


    st.markdown(
        '<p class="main-header">🔮 Prédiction d\'un Nouveau Dossier</p>',
        unsafe_allow_html=True
    )


    st.markdown(
        "Renseignez les informations du demandeur puis cliquez sur **Prédire**."
    )


    with st.form("prediction_form"):


        st.subheader("Informations personnelles")


        col1,col2,col3 = st.columns(3)


        with col1:

            gender = st.selectbox(
                "Genre",
                ["Male","Female"]
            )


            married = st.selectbox(
                "Marié(e)",
                ["Yes","No"]
            )


            dependents = st.selectbox(
                "Personnes à charge",
                ["0","1","2","3+"]
            )



        with col2:


            education = st.selectbox(
                "Niveau d'étude",
                ["Graduate","Not Graduate"]
            )


            self_employed = st.selectbox(
                "Travailleur indépendant",
                ["No","Yes"]
            )


            property_area = st.selectbox(
                "Zone",
                ["Urban","Semiurban","Rural"]
            )



        with col3:


            credit_history = st.selectbox(
                "Historique crédit",
                ["Oui","Non"]
            )


            loan_term = st.selectbox(
                "Durée du prêt",
                [360,180,120,84,300,60,36]
            )



        st.divider()


        st.subheader("Informations financières")


        c1,c2,c3 = st.columns(3)


        with c1:

            applicant_income = st.number_input(
                "Revenu demandeur",
                min_value=0,
                value=5000
            )


        with c2:

            coapplicant_income = st.number_input(
                "Revenu co-demandeur",
                min_value=0,
                value=0
            )


        with c3:

            loan_amount = st.number_input(
                "Montant du prêt",
                min_value=0,
                value=150
            )

        # Vérification des champs obligatoires
        formulaire_valide = (
        gender != "-- Sélectionner --"
        and married != "-- Sélectionner --"
        and dependents != "-- Sélectionner --"
        and education != "-- Sélectionner --"
        and self_employed != "-- Sélectionner --"
        and property_area != "-- Sélectionner --"
        and credit_history != "-- Sélectionner --"
        and applicant_income > 0
        and loan_amount > 0
        )

        submitted = st.form_submit_button(
            "Valider la Prédiction" ,
            disabled=not formulaire_valide
        )



    if submitted:

        # ==============================
        # CREATION DES DONNEES CLIENT
        # ==============================

        new_data = pd.DataFrame([{

            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": 1.0 if credit_history == "Oui" else 0.0,
            "Property_Area": property_area

        }])


        var_num = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History"
        ]


        var_cat = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area"
        ]


        # ==============================
        # PRETRAITEMENT
        # ==============================

        new_data[var_num] = imputer_num.transform(
            new_data[var_num]
        )


        new_data[var_cat] = imputer_cat.transform(
            new_data[var_cat]
        )


        X_num = scaler.transform(
            new_data[var_num]
        )


        X_cat = encoder.transform(
            new_data[var_cat]
        )


        X_final = np.hstack(
            (
                X_num,
                X_cat
            )
        )


        # ==============================
        # PREDICTION
        # ==============================

        prediction = model.predict(
            X_final
        )[0]


        probability = model.predict_proba(
            X_final
        )[0]


        # ==============================
        # ENREGISTREMENT HISTORIQUE
        # ==============================

        nouvelle_demande = pd.DataFrame([{

            "Date": pd.Timestamp.now().strftime(
                "%d/%m/%Y %H:%M"
            ),

            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,

            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,

            "LoanAmount": loan_amount,

            "Loan_Amount_Term": loan_term,

            "Credit_History": credit_history,

            "Property_Area": property_area,

            "Decision":
                "Approuvé" if prediction == 1 else "Refusé",

            "Probabilité approbation":
                round(probability[1] * 100, 2),

            "Confiance":
                round(max(probability) * 100, 2)

        }])


        st.session_state.historique_predictions = pd.concat(
            [
                st.session_state.historique_predictions,
                nouvelle_demande
            ],
            ignore_index=True
        )


        # ==============================
        # AFFICHAGE RESULTAT
        # ==============================

        st.divider()

        st.subheader(
            "Résultat de la prédiction"
        )


        col1, col2 = st.columns(2)


        with col1:

            if prediction == 1:

                st.success(
                    "##  PRÊT APPROUVÉ"
                )

            else:

                st.error(
                    "##  PRÊT REFUSÉ"
                )


            st.metric(
                "Confiance du modèle",
                f"{max(probability)*100:.2f}%"
            )


            st.metric(
                "Probabilité approbation",
                f"{probability[1]*100:.2f}%"
            )


            st.metric(
                "Probabilité refus",
                f"{probability[0]*100:.2f}%"
            )


        with col2:

            fig = go.Figure(
                go.Indicator(

                    mode="gauge+number",

                    value=probability[1]*100,

                    title={
                        "text":
                        "Probabilité d'approbation"
                    },

                    gauge={
                        "axis":{
                            "range":[0,100]
                        }
                    }
                )
            )


            fig.update_layout(
                height=350
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        st.success(
            "✅ La demande a été enregistrée dans l'historique."
        )


        if st.button("🔄 Nouvelle prédiction"):

            st.rerun()

# ----------------------------------------------------------------------------
# PAGE 5 - HISTORIQUE DES PREDICTIONS
# ----------------------------------------------------------------------------

elif page == "📋 Historique des prédictions":

    st.markdown(
        '<p class="main-header">📋 Historique des Demandes</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Liste des dossiers ayant effectué une prédiction."
    )


    historique = st.session_state.historique_predictions


    if historique.empty:

        st.info(
            "Aucune prédiction enregistrée pour le moment."
        )

    else:

        # Nombre total de demandes

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Nombre de demandes",
                len(historique)
            )


        with col2:
            taux = (
                historique["Decision"]
                .value_counts()
                .get("Approuvé",0)
            )

            st.metric(
                "Prêts approuvés",
                taux
            )


        st.divider()


        # Tableau historique

        st.subheader(
            "📄 Liste des prédictions"
        )


        st.dataframe(
            historique,
            use_container_width=True,
            hide_index=True
        )


        st.divider()


        # Export CSV

        csv = historique.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="Télécharger l'historique CSV",
            data=csv,
            file_name="historique_predictions.csv",
            mime="text/csv"
        )


        # Suppression historique

        if st.button(
            " Supprimer l'historique"
        ):

            st.session_state.historique_predictions = pd.DataFrame(
                columns=historique.columns
            )

            st.success(
                "Historique supprimé."
            )

            st.rerun()