
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Terör Saldırıları Analizi", layout="wide")
st.title("Orta Doğu ve Kuzey Afrika Bölgesindeki Terör Saldırıları: Türkiye'yi Etkileyen Olası Tehditler")

# Veri tabanı seçimi
db_options = ["Select Database", "MENA", "EU", "Asia"]
selected_db = st.sidebar.selectbox("Veri Tabanı Seç", db_options)

def load_dataset(name):
    if name == "MENA":
        return pd.read_csv("first_100_rows.csv")
    elif name == "EU":
        return pd.read_csv("first_20000_rows.csv")
    elif name == "Asia":
        return pd.read_csv("asia_dataset.csv")
    return None

df = None
if selected_db != "Select Database":
    df = load_dataset(selected_db)
    st.success(f"✅ Yüklendi: {selected_db}")
    st.dataframe(df)

# Session state için başlangıç
if "results" not in st.session_state:
    st.session_state["results"] = {}  # Algoritma adı -> sonuç dict
if "trained_models" not in st.session_state:
    st.session_state["trained_models"] = {}  # Algoritma adı -> model objesi
if "train_columns" not in st.session_state:
    st.session_state["train_columns"] = {}  # Algoritma adı -> train kolonları

# Sorgular (koşullar)
st.sidebar.markdown("### 🧮 Sorgular (Koşullar)")
query_dict = {}
if df is not None:
    all_columns = df.columns.tolist()
    if "target_column" not in st.session_state:
        st.session_state["target_column"] = None

    st.session_state["target_column"] = st.sidebar.selectbox("🎯 Hedef sütun", all_columns)
    feature_columns = [col for col in all_columns if col != st.session_state["target_column"]]
    max_conditions = len(feature_columns)
    condition_count = st.sidebar.slider("Kaç adet koşul kullanmak istiyorsunuz?", 1, max_conditions, 1)

    for i in range(condition_count):
        col = st.sidebar.selectbox(f"{i+1}. Kolon seçin", feature_columns, key=f"cond_col_{i}")
        val = st.sidebar.text_input(f"Değer girin ({col})", key=f"cond_val_{i}")
        if val != "":
            query_dict[col] = val
    st.sidebar.markdown("---")

# Model seçimi ve eğitimi
if df is not None and st.session_state["target_column"]:
    features = df.drop(columns=[st.session_state["target_column"]])
    labels = df[st.session_state["target_column"]]
    features = pd.get_dummies(features, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    algo = st.selectbox("🧠 Algoritma seç", ["SVM", "Decision Tree", "KNN"])

    if st.button("🏁 Eğit ve Değerlendir"):
        if algo == "SVM":
            model = SVC()
        elif algo == "Decision Tree":
            model = DecisionTreeClassifier()
        else:
            model = KNeighborsClassifier()

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)

        # Sonuçları ve modeli kaydet
        st.session_state["results"][algo] = {
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "y_test": y_test,
            "y_pred_test": y_pred_test
        }
        st.session_state["trained_models"][algo] = model
        st.session_state["train_columns"][algo] = X_train.columns.tolist()

        st.success(f"🎯 {algo} Modeli Eğitildi ve Değerlendirildi")

# Sonuçları ve grafiklerini ayrı ayrı göster
if st.session_state["results"]:
    st.header("📊 Algoritma Sonuçları ve Başarı Grafikleri")
    for algo, metrics in st.session_state["results"].items():
        with st.expander(f"📌 {algo} Sonuçları ve Grafikler"):
            st.subheader(f"{algo} Performans Metrikleri")
            st.write({k: v for k, v in metrics.items() if k not in ["y_test", "y_pred_test"]})

            # Confusion matrix çizimi
            cm_fig, ax = plt.subplots()
            cm = pd.crosstab(metrics["y_test"], metrics["y_pred_test"])
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            ax.set_title(f"{algo} - Confusion Matrix")
            st.pyplot(cm_fig)

            # Accuracy Bar Grafiği (Train vs Test)
            acc_fig, ax2 = plt.subplots()
            ax2.bar(["Train Accuracy", "Test Accuracy"], [metrics["Train Accuracy"], metrics["Test Accuracy"]], color=["green", "red"])
            ax2.set_ylim(0, 1)
            ax2.set_title(f"{algo} - Doğruluk (Accuracy)")
            st.pyplot(acc_fig)

            # Diğer metrikler bar grafiği
            metric_fig, ax3 = plt.subplots()
            ax3.bar(["Precision", "Recall", "F1-Score"], [metrics["Precision"], metrics["Recall"], metrics["F1-Score"]], color=["blue", "orange", "purple"])
            ax3.set_ylim(0, 1)
            ax3.set_title(f"{algo} - Diğer Metrikler")
            st.pyplot(metric_fig)

# Tahmin işlemi
if query_dict and st.session_state.get("trained_models"):
    st.header("🔮 Tahmin")

    # Kullanıcı seçimine göre modeli al
    selected_model = st.selectbox("Tahmin için algoritma seç", list(st.session_state["trained_models"].keys()))

    if st.button("🔍 Tahmin Et"):
        input_df = pd.DataFrame([query_dict])
        input_encoded = pd.get_dummies(input_df)

        # Modelin eğitimde kullandığı kolonları tamamla
        for col in st.session_state["train_columns"][selected_model]:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[st.session_state["train_columns"][selected_model]]

        prediction = st.session_state["trained_models"][selected_model].predict(input_encoded)[0]
        st.success(f"Bu koşullara göre tahmin edilen sonuç: **{prediction}**")
