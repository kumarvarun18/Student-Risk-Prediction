import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

with open("assests/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Page Configuration
st.set_page_config(
    page_title="Student Performance Risk Prediction",
    page_icon="🎓",
    layout="wide"
)
# Load Model
model=joblib.load("model/student_risk_model.pkl")
preprocessor=joblib.load("model/preprocessor.pkl")
# sidebar
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Page",
    [   "Home",
        "Dashboard",
        "Prediction",
        "Model Performance",
        "About"
    ]
)
# Home Page
if menu == "Home":
    st.markdown(""" 
            <div class="hero">
            <h1>🎓 Student Performance Risk Prediction System</h1>
            <p>An AI-powered Machine Learning Web application for early identification of  whether a student is at academic risk.</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
    # Project Overview
    st.markdown("""
            <div class="custom-card">
                <h3> Project Overview</h3>
                <p>
                The Student Performance Risk Prediction System is a MAchine Learning application that predicts whether a student is academically at risk
                based on academic and behavioural attributes such as GPA, attendance, assignment submission, lab performance, online learning hours,
                forum participation, and socio-economic status.</p>
            </div>
            """,unsafe_allow_html=True)
    st.markdown("## Project Objectives")
    col1,col2 = st.columns(2)

    with col1:
        st.success("""
            - Predict academic risk
            - Enable early intervention
            - Improve student success
            - Support educators""")
    with col2:
        st.info("""
            - Analyze student data
            - Machine Learning Prediction
            - Interactive Dashboard
            - Streamlit Web Application""")
    st.markdown("---")
    # Technology Stack
    st.subheader("Technology Stack")
    tech1, tech2, tech3, tech4 = st.columns(4)
    tech1.metric("Language", "Python")
    tech2.metric("Framework", "Streamlit")
    tech3.metric("ML Library", "Scikit-learn")
    tech4.metric("Visualization", "Plotly")
    st.markdown("---")
    # Workflow
    st.subheader("Project Workflow")
    st.markdown("""
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px">

        <div class="workflow-card">Dataset</div>

        <div class="workflow-card">EDA</div>

        <div class="workflow-card">Preprocessing</div>

        <div class="workflow-card">Model Training</div>

        <div class="workflow-card">Hyperparameter Tuning</div>

        <div class="workflow-card">Deployment</div>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Features
    st.subheader("Key Features")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("""
            - Interactive Dashboard 
            - Data Visualization 
            - Dataset Insights""")
    with c2:
        st.success("""
            - Risk Prediction
            - Probability Score
            - Recommendations
            """)
    with c3:
        st.success("""
            - Model Performance
            - Reports
            - Project Information
            """)
    st.markdown("---")
    # Quick Statistics
    st.subheader("Project Statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Features", "10")
    s2.metric("Target", "Binary")
    s3.metric("Algorithm", "Gradient Boosting")
    s4.metric("Deployment", "Streamlit")
    st.markdown("---")
    # Why this project
    st.markdown("""
    <div class="highlight-box">

    <h2>Why This Project?</h2>

    <p>
        Educational institutions often face challenges in identifying students
        who require additional academic support. This application uses Machine
        Learning to analyze student information and provide early predictions,
        helping educators make informed decisions and improve student outcomes.
    </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.success("Navigate through the sidebar to explore the Dashboard, EDA Insights, Prediction, Model Performance, and About pages.")
# About Page
elif menu == "About":
    st.header("About Project")
    st.markdown("""
    ## 🎓 Student Performance Risk Prediction System

    This project is a **Machine Learning-based Binary Classification System**
    developed to identify students who may be academically at risk.
    It analyzes academic and behavioural factors and predicts whether a student
    requires early academic intervention.

    The application was developed using **Python**, **Scikit-learn**, and
    **Streamlit** as part of a **45-Day Industrial Training Program**.
    """)

    st.divider()
    st.subheader("Features Used")
    features= [
        "Age",
        "Gender",
        "Socio Economic Status",
        "Attendance Rate",
        "Assignments Submitted",
        "Online Learning Hours",
        "Forum Participation",
        "Lab Performance",
        "Extracurricular Participation",
        ]
    for feature in features:
        st.markdown(f"✅ {feature}")

    st.subheader("Project Objectives")

    st.markdown("""
    - Predict students who are academically at risk.
    - Support early intervention using data-driven insights.
    - Provide an interactive dashboard for data exploration.
    - Deploy the trained machine learning model using Streamlit.
    """)
    st.divider()
    st.subheader("Technologies Used")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Language", "Python")
    col2.metric("Framework", "Streamlit")
    col3.metric("ML Model", "Gradient Boosting")
    col4.metric("Library", "Scikit-learn")
    st.divider()
    st.header("Features")
    st.success("""
    - Interactive Dashboard
    - Dataset Filtering
    - Student Risk Prediction
    - Probability Score
    - Model Performance
    - Responsive User Interface
    """)
    st.divider()
    st.subheader("Project Workflow")
    st.markdown("""
    **Dataset → Data Exploration → EDA → Data Preprocessing → Model Training → Hyperparameter Tuning → Model Evaluation → Streamlit Deployment**
    """)
    st.divider()
    st.subheader("Machine Learning Details")
    st.markdown("""
    - **Learning Type:** Supervised Learning
    - **Problem Type:** Binary Classification
    - **Final Model:** Gradient Boosting Classifier
    - **Target Variable:** At_Risk
    """)
    st.divider()
    st.subheader("Future Enhancements")
    st.markdown("""
    - Database integration
    - User authentication
    - Real-time student monitoring
    - Downloadable prediction reports
    - Personalized academic recommendations
    """)
    st.divider()
    st.info(
        "This project demonstrates how Machine Learning can support educators "
        "in identifying academically at-risk students and enabling timely intervention."
    )

elif menu == "Dashboard":
    st.header("Student Dataset Dashboard")
   # Load Dataset
    df=pd.read_csv("data/risk_students (3).csv")

    # Main Metrics
    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric("Total Students", df.shape[0])
    with col2:
        st.metric("Total Features",df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    st.divider()
    # Dataset Information
    st.subheader("Dataset Information")
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        numerical = len(df.select_dtypes(include="number").columns)
        st.metric("Numerical Features", numerical)
    with col4:
        categorical = len(df.select_dtypes(include="object").columns)
        st.metric("Categorical Features", categorical)
    st.divider()

    # Average Statistics
    st.subheader("Average Statistics")
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.metric("Average GPA", round(df["GPA"].mean(),2))
    with col2:
        st.metric("Average Attendance", round(df["Attendance_Rate"].mean(),2))
    with col3:
        st.metric("Average Learning  Hours", round(df["Online_Learning_Hours"].mean(),2))
    with col4:
        st.metric("Average Lab Performance", round(df["Lab_Performance"].mean(),2))
    st.divider()

    # Dataset Preview
    st.subheader("Dataser Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.divider()

    # Filter Dataset
    st.subheader("Filter Dataset")
    filtered_df = df.copy()
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["All"] + sorted(df["Gender"].unique().tolist()))
    with col2:
        socio_options = ["All", "Low", "High","Medium"]
        socio = st.selectbox("Socio Economic Status", socio_options)
    with col3:
        risk = st.selectbox("Risk Status", ["All"] + sorted(df["At_Risk"].unique().tolist()))
    col4, col5, col6 = st.columns(3)
    with col4:
        gpa_range  = st.slider("GPA Range", float(df["GPA"].min()),float(df["GPA"].max()),(float(df["GPA"].min()), float(df["GPA"].max())))
    with col5:
        age = st.slider("Minimum Age", int(df["Age"].min()),int(df["Age"].max()), int(df["Age"].max()))
    with col6:
        attendance = st.slider("Minimum Attendance", 0,100,0)
    # Apply Filter
    if gender != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == gender]
    if socio != "All":
        filtered_df = filtered_df[filtered_df["Socio_Economic_Status"] == socio]
    if risk != "All":
        filtered_df = filtered_df[filtered_df["At_Risk"] == risk]
    filtered_df = filtered_df[(filtered_df["GPA"] >= gpa_range[0]) & (filtered_df["GPA"] <= gpa_range[1])]
    filtered_df = filtered_df[filtered_df["Age"] >= age]
    filtered_df = filtered_df[(filtered_df["Attendance_Rate"] >= attendance)]

    st.success(f"Showing {len(filtered_df)} students(s) after applying filters.")
    st.dataframe(filtered_df, use_container_width=True)
    st.divider()

    # Target Distribution
    st.subheader("Student Risk Distribution")
    target_count = df["At_Risk"].value_counts()

    fig =px.pie(values= target_count.values, names=["Not At Risk", "At Risk"],
                title="Risk Distribution",
                hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    # GPA Distribution
    st.subheader("GPA Distribution")
    fig2=px.histogram(df,x="GPA",title="GPA Distribution", nbins=20)
    st.plotly_chart(fig2,use_container_width=True)

    # Attendance Distribution
    st.subheader("Attendance Analysis")
    fig3=px.histogram(df,x="Attendance_Rate", title="Attendance Rate Distribution")
    st.plotly_chart(fig3, use_container_width=True)

    # Gender Distribution
    st.subheader("Gender Distribution")
    fig4=px.bar(df["Gender"].value_counts().reset_index(),x="Gender",y="count",title="Gender Distribution")
    st.plotly_chart(fig4, use_container_width=True)

    # GPA vs Attendance
    st.subheader("GPA vs Attendance")
    fig5=px.scatter(df,x="Attendance_Rate",y="GPA",color="At_Risk",title="Attendance vs GPA", hover_data=["Age", "Gender"])
    st.plotly_chart(fig5,use_container_width=True)
    st.divider()

    # Correlation Heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["number"])
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(),annot=True,cmap="Blues",linewidths=0.5,ax=ax)
    st.pyplot(fig)
    st.divider()
    # Data summary
    st.subheader("Dataset Summary")
    st.dataframe(df.describe(), use_container_width=True)
    st.divider()

    # Top 10 Students by GPA
    st.subheader("Top 10 Students by GPA")
    top_students = df.sort_values(by="GPA", ascending=False).head(10)
    st.dataframe(top_students, use_container_width=True)
    st.divider()

    # Risk Summary
    st.subheader("Risk Summary")
    risk_summary = (df["At_Risk"].value_counts().reset_index())
    risk_summary.columns=["Risk Status", "Count"]
    st.dataframe(risk_summary, use_container_width=True)
    st.divider()

    # Key Insights
    st.subheader("Key Insights")
    st.success("""
            - Students with higher GPA generally have a lower academic risk.
            - Better attendance is associated with improved academic performance.
            - Assignment Submission and lab performance are important indicators of success.
            - Students with low GPA and poor attendance are more likely to be classified as AT Risk.
            - Online learning hours and forum participation also contribute to prediction.
            """)
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <center>
        <h4> Student Performancr Risk Prediction System</h4>
        <p>Devolped using <b>Streamlit</b> &bull; <b>Python</b>  &bull; <b>Scikit-learn</b>&bull;  <b>Plotly</b></p>
        </center>
        """,
        unsafe_allow_html=True
    )
elif menu == "Prediction":
    st.header("Student Risk Prediction")
    st.write("""Enter the student's details below to predict whether the student is academically at risk.""")
    st.divider() 
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        # Student Information
        with col1:
            st.subheader("Student Information")
            age = st.number_input("Age", min_value=15, max_value=30, value=20)
            gender = st.selectbox("Gender", ["Male", "Female"])
            socio = st.selectbox("Socio Ecpnomic Status", ["Low", "Medium", "High"])
        # Academic Information
        with col2:
            st.subheader("Academic Information")
            gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value = 2.5, step=0.1)
            attendance = st.slider("Attendance Rate", 0, 100, 75)
            assignments = st.slider("Assignments Submitted", 0, 100, 70)
            learning = st.slider("Online Learning Hours", 0, 20, 8)
            forum = st.slider("Forum Participation", 0 ,20, 5)
            lab = st.slider("Lab Performance", 0, 100, 75)
            extracurricular = st.selectbox("Extracurricular Participation", ["Yes","No"])
            extracurricular_value = 1 if extracurricular =="Yes" else 0
            st.divider()
            submitted = st.form_submit_button("Predict Student Risk", use_container_width=True)
        if submitted:
            # Create Input Dataframe
            input_df = pd.DataFrame({"Age":[age],"Gender":[gender],"Socio_Economic_Status":[socio],
                                    "GPA": [gpa], "Attendance_Rate":[attendance], "Assignments_Submitted":[assignments],"Online_Learning_Hours":[learning], "Forum_Participation": [forum],
                                    "Lab_Performance": [lab], "Extracurricular_Participation": [extracurricular_value]})
            
            processed_data = preprocessor.transform(input_df)
            prediction = model.predict(processed_data)
            probability = model.predict_proba(processed_data)
            st.divider()
            st.subheader("Prediction Result")
            risk_probability = probability[0][1]*100
            safe_probability = probability[0][0]*100
            if prediction[0] == 1:
                st.error("Student is At Academic Risk")
                st.progress(min(risk_probability / 100, 1.0))
                st.metric("Risk Probability", f"{risk_probability:.2f}%")
                st.warning("""
                           Recommendations 
                        - Improve attendance.
                        - Submit Assignmnets on time.
                        - Increase online learning hours.
                        - Participate more in laboratory activities.
                        - Engage in classroom discussions and forum participation.
                        - seek academic guidance from mentors.
                        """)
            else: 
                st.success("Students is Not At Academic Risk")
                st.progress(min(safe_probability/100, 1.0))
                st.metric("Prediction Confidence",f"{safe_probability:.2f}%")
                st.info("""
                        - Keep Up the Good Work
                        - Maintain regular attendace.
                        - Continue submitting assignments on time.
                        - Keep your GPA consistent.
                        - Participate actively in lab sessions.
                        - Continue engaging in extracurricular activities.""")
            st.divider()
            st.success("Prediction completed successfully using the trained Gradient Boosting Model")
elif menu == "Model Performance":
    st.header("Model Performance")
    st.write("""This page presents the evaluation results of the final **Gradient Boosting Classifier** after hyperparameter tuning.""")
    st.divider()
    st.header("Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", "80.75%")
    with col2:
        st.metric("Precision", "72%")
    with col3:
        st.metric("Recall", "54%")
    with col4:
        st.metric("F1 Score", "62%")
    with col5:
        st.metric("ROC-AUC", "0.8215")
    st.divider()
    st.subheader("Confusion Matrix")
    st.image("images/Confusion Matrix.png", use_container_width = True, caption="Confusion Matrix")
    st.divider()
    st.subheader("Classification Report")
    report = pd.DataFrame({"Class":["Not At Risk", "At Risk"],"Precision":["0.83","0.72"], "Recall":["0.92","0.54"],"F1 Score":["0.87","0.62"], "Support":["285","115"]})
    st.dataframe(report, use_container_width=True)
    st.divider()
    # Feature Importance
    # st.subheader("Feature Importance")
    st.subheader("Model Interpretation")
    st.success("""
               - The final Gradient Boosting Classifier acheived strong predictive performance.
               - The model successfully distinguish between students who are academically at risk and those who are not.
               - Academic indicators such as GPA, Attendance Rate, Assignment Submission, and LAb Performance play an important role in prediction
               - This model can support educators by enabling early indentification of students who may require addtional academic support.
               """)
    st.divider()
    st.subheader("Strengths")
    st.markdown("""
                - High predictable accuracy.
                - Robust ensemble learning algoirthm.
                - Handles nonnlinear relationships effectively.
                - Suitable for binary classification tasks.
                - Supports data-driven academic tasks.
                - Easy to deploy using Streamlit.""")
    st.divider()
    st.subheader("Limitations")
    st.markdown("""
                - Performance depends on the quality of the training dataset.
                - Predictions are limited to the selected input features.
                - The model should support, not replace, academic decision-making.
                - Additional student-related factors may further improve prediction performance.""")
    st.divider()
    st.subheader("Future Scope")
    st.markdown("""
                - Integrate with a student information system.
                - Enable real-time prediction usign live student data.
                - Add download PDF reports.
                - Connect the application to a database.
                - Include personalized academic recommendations and progress tracking.""")
    st.divider()
    st.subheader("Conclusion")
    st.info("""The **Gradient Boosting Classifier** was selected as the final model after evaluation.Its performance demonstrates that machine learning can effectively support early identification of academically at-risk students. This Streamlit application combines predictive analystics with an interactive interface to assist educators in making informed decisions.""")

   