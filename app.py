import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(
    page_title="Iris AI Classifier",
    page_icon="🌸",
    layout="wide"
)

st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #0f172a
    );
    color:white;
}

/* Hide Streamlit Elements */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Title */
.main-title{
    text-align:center;
    font-size:60px;
    font-weight:800;
    color:white !important;
}       

/* Subtitle */
.subtitle{
    text-align:center;
    font-size:18px;
    color:#cbd5e1;
    margin-bottom:30px;
}

/* Glass Cards */
.glass{
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(15px);

    border:1px solid rgba(255,255,255,0.15);

    padding:25px;
    border-radius:20px;

    box-shadow:0px 8px 32px rgba(0,0,0,0.3);
}

/* Section Headings */
h1,h2,h3,h4,h5,h6{
    color:white !important;
}

/* Button */
.stButton > button{

    width:100%;
    height:60px;

    background: linear-gradient(
        90deg,
        #00E5FF,
        #7C4DFF
    );

    color:white;
    font-size:20px;
    font-weight:bold;

    border:none;
    border-radius:15px;
}

.stButton > button:hover{
    transform:scale(1.02);
    transition:0.3s;
}

/* Result Box */
.result-box{
    background:rgba(0,255,153,0.12);
    border:1px solid #00ff99;

    padding:25px;
    border-radius:20px;

    text-align:center;

    font-size:30px;
    font-weight:bold;

    color:#00ff99;
}

/* Metrics */
div[data-testid="metric-container"] *{
    color:white !important;
}

    border:1px solid rgba(255,255,255,0.15);

    padding:15px;
    border-radius:15px;
}

/* Dataframe */
[data-testid="stDataFrame"]{
    border-radius:15px;
}

/* Make all text white */
html, body, [class*="css"] {
    color: white !important;
}

p, span, div, label {
    color: white !important;
}

 h2, h3, h4, h5, h6 {
    color: white !important;
}

/* Slider labels */
label {
    color: white !important;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


st.markdown("""
<h1 class='main-title'>
🌸 IRIS FLOWER CLASSIFIER
</h1>

<p class='subtitle'>
Predict Iris Flower Species Using Machine Learning
</p>
""", unsafe_allow_html=True)


st.markdown("""
<div class='glass'>
<h3 style='text-align:center;'>
❀ Flower Prediction Dashboard ❀
</h3>

<p style='text-align:center;color:#cbd5e1'>
Enter flower measurements and instantly predict
whether the flower belongs to Setosa,
Versicolor, or Virginica.
</p>
</div>
""", unsafe_allow_html=True)

st.write("")


left, right = st.columns(2)


with left:

    st.markdown("<div class='glass'>",
                unsafe_allow_html=True)

    st.subheader("🌿 Flower Measurements")

    sepal_length = st.slider(
        "Sepal Length (cm)",
        4.0, 8.0, 5.8
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        2.0, 4.5, 3.0
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        1.0, 7.0, 4.3
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        0.1, 2.5, 1.3
    )

    st.markdown("</div>",
                unsafe_allow_html=True)


with right:

    st.markdown("<div class='glass'>",
                unsafe_allow_html=True)

    st.subheader("📊 Live Input Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Sepal Length",
            f"{sepal_length} cm"
        )

        st.metric(
            "Petal Length",
            f"{petal_length} cm"
        )

    with c2:
        st.metric(
            "Sepal Width",
            f"{sepal_width} cm"
        )

        st.metric(
            "Petal Width",
            f"{petal_width} cm"
        )

    st.markdown("</div>",
                unsafe_allow_html=True)

st.write("")


if st.button("🔍 Predict Species"):

    prediction = model.predict([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    probs = model.predict_proba([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    species = iris.target_names[
        prediction[0]
    ]

    st.markdown(
        f"""
        <div class='result-box'>
        🌸 Predicted Species<br><br>
        {species.upper()}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Probability Chart
    st.markdown("<div class='glass'>",
                unsafe_allow_html=True)

    st.subheader("📈 Prediction Confidence")

    confidence_df = pd.DataFrame({
        "Species": iris.target_names,
        "Probability": probs[0]
    })

    st.bar_chart(
        confidence_df.set_index(
            "Species"
        )
    )

    st.markdown("</div>",
                unsafe_allow_html=True)

    st.write("")

    # Probability Metrics
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "🌸 Setosa",
            f"{probs[0][0]*100:.2f}%"
        )

    with m2:
        st.metric(
            "🌿 Versicolor",
            f"{probs[0][1]*100:.2f}%"
        )

    with m3:
        st.metric(
            "🌺 Virginica",
            f"{probs[0][2]*100:.2f}%"
        )


st.write("")
st.write("")

st.markdown("""
<hr>

""", unsafe_allow_html=True)