import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sys
import pandas as pd
from io import BytesIO
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
st.set_page_config(page_title="Health Insurance Cost Predictor", layout="wide", page_icon="🏥")
sys.modules['sklearn.ensemble.gradient_boosting'] = sys.modules['sklearn.ensemble']
sys.modules['sklearn.ensemble._gb'] = sys.modules['sklearn.ensemble']
try:
    if os.path.exists('Insuarance(gbr).pkl'):
        with open('Insuarance(gbr).pkl', 'rb') as f:
            model = pickle.load(f)
        model_loaded = True
    else:
        model_loaded = False
        model = None
except Exception:
    model_loaded = False
    model = None
np.random.seed(42)
db_size = 5000
database = pd.DataFrame({
    'age': np.random.randint(18, 65, db_size),
    'sex': np.random.choice(['male', 'female'], db_size),
    'bmi': np.random.normal(28, 6, db_size).clip(15, 50),
    'children': np.random.choice([0, 1, 2, 3, 4, 5], db_size, p=[0.3, 0.25, 0.25, 0.1, 0.07, 0.03]),
    'smoker': np.random.choice(['no', 'yes'], db_size, p=[0.8, 0.2]),
    'region': np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], db_size)
})
le_sex = LabelEncoder()
le_sex.fit(['female', 'male'])
le_smoker = LabelEncoder()
le_smoker.fit(['no', 'yes'])
le_region = LabelEncoder()
le_region.fit(['northeast', 'northwest', 'southeast', 'southwest'])
database['sex_encoded'] = le_sex.transform(database['sex'])
database['smoker_encoded'] = le_smoker.transform(database['smoker'])
database['region_encoded'] = le_region.transform(database['region'])
def calculate_cost(age, sex_enc, bmi, children, smoker_enc, region_enc):
    base_cost = 3000
    age_factor = (age - 18) * 240
    sex_factor = sex_enc * 131.3
    bmi_factor = max(0, (bmi - 18.5)) * 393
    children_factor = children * 475.5
    smoker_factor = smoker_enc * 23847.5
    region_factor = region_enc * 352.9
    total = base_cost + age_factor + sex_factor + bmi_factor + children_factor + smoker_factor + region_factor
    return max(1121.87, min(63770.43, total))
X_db = database[['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']].values
if model is not None:
    database['predicted_cost'] = model.predict(X_db)
else:
    database['predicted_cost'] = [calculate_cost(row[0], row[1], row[2], row[3], row[4], row[5]) for row in X_db]
USD_TO_INR = 83.5
@st.cache_data
def train_and_evaluate_models():
    X = database[['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']].values
    y = database['predicted_cost'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100
        results[name] = {"model": m, "y_test": y_test, "y_pred": y_pred, "R2": round(r2, 4), "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MSE": round(mse, 2), "MAPE": round(mape, 2)}
    return results
def create_comparison_charts(user_data, prediction_usd, database):
    fig = make_subplots(rows=2, cols=3, subplot_titles=('Your Cost vs Average', 'Cost by Age Group', 'Cost by BMI Category', 'Cost by Smoking Status', 'Cost by Children', 'Cost by Region'), specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]])
    avg_cost_overall = database['predicted_cost'].mean()
    fig.add_trace(go.Bar(x=['Your Cost', 'Average Cost'], y=[prediction_usd, avg_cost_overall], marker_color=['#FF6B6B', '#4ECDC4'], text=[f'${prediction_usd:,.0f}', f'${avg_cost_overall:,.0f}'], textposition='outside'), row=1, col=1)
    age_groups = pd.cut(database['age'], bins=[0, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '51+'])
    age_avg = database.groupby(age_groups)['predicted_cost'].mean()
    user_age_group = pd.cut([user_data['age']], bins=[0, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '51+'])[0]
    fig.add_trace(go.Bar(x=age_avg.index.astype(str), y=age_avg.values, marker_color=['#FF6B6B' if g == user_age_group else '#95E1D3' for g in age_avg.index]), row=1, col=2)
    fig.add_hline(y=prediction_usd, line_dash="dash", line_color="red", row=1, col=2)
    bmi_groups = pd.cut(database['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    bmi_avg = database.groupby(bmi_groups)['predicted_cost'].mean()
    user_bmi_group = pd.cut([user_data['bmi']], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])[0]
    fig.add_trace(go.Bar(x=bmi_avg.index.astype(str), y=bmi_avg.values, marker_color=['#FF6B6B' if g == user_bmi_group else '#F38181' for g in bmi_avg.index]), row=1, col=3)
    fig.add_hline(y=prediction_usd, line_dash="dash", line_color="red", row=1, col=3)
    smoker_avg = database.groupby('smoker')['predicted_cost'].mean()
    fig.add_trace(go.Bar(x=smoker_avg.index, y=smoker_avg.values, marker_color=['#FF6B6B' if s == user_data['smoker'].lower() else '#AAF683' for s in smoker_avg.index]), row=2, col=1)
    fig.add_hline(y=prediction_usd, line_dash="dash", line_color="red", row=2, col=1)
    children_avg = database.groupby('children')['predicted_cost'].mean()
    fig.add_trace(go.Bar(x=children_avg.index.astype(str), y=children_avg.values, marker_color=['#FF6B6B' if c == user_data['children'] else '#FFEAA7' for c in children_avg.index]), row=2, col=2)
    fig.add_hline(y=prediction_usd, line_dash="dash", line_color="red", row=2, col=2)
    region_avg = database.groupby('region')['predicted_cost'].mean()
    fig.add_trace(go.Bar(x=region_avg.index, y=region_avg.values, marker_color=['#FF6B6B' if r == user_data['region'].lower() else '#DFE6E9' for r in region_avg.index]), row=2, col=3)
    fig.add_hline(y=prediction_usd, line_dash="dash", line_color="red", row=2, col=3)
    fig.update_layout(height=580, showlegend=False, title_text="Cost Comparison Analysis", margin=dict(t=60, b=20))
    fig.update_yaxes(title_text="Cost (USD)")
    return fig
def generate_recommendations(user_data, prediction_usd, database):
    recommendations = []
    avg_cost = database['predicted_cost'].mean()
    cost_diff_pct = ((prediction_usd - avg_cost) / avg_cost) * 100
    if user_data['smoker'].lower() == 'yes':
        non_smoker_similar = database[(database['smoker'] == 'no') & (database['age'].between(user_data['age']-5, user_data['age']+5)) & (database['bmi'].between(user_data['bmi']-2, user_data['bmi']+2))]['predicted_cost'].mean()
        potential_savings = prediction_usd - non_smoker_similar
        recommendations.append({'category': 'Smoking Cessation', 'priority': 'HIGH', 'impact': f'Potential savings: ${potential_savings:,.0f}/year (₹{potential_savings*USD_TO_INR:,.0f})', 'action': 'Quit smoking to reduce insurance costs by 50-70%. Join smoking cessation programs, use nicotine replacement therapy, or consult a healthcare provider.', 'timeframe': '6-12 months to see premium reductions'})
    if user_data['bmi'] > 30:
        normal_bmi_similar = database[(database['bmi'].between(18.5, 25)) & (database['age'].between(user_data['age']-5, user_data['age']+5)) & (database['smoker'] == user_data['smoker'].lower())]['predicted_cost'].mean()
        potential_savings = prediction_usd - normal_bmi_similar
        target_bmi = 24.9
        current_weight_kg = user_data['bmi'] * (1.7 ** 2)
        target_weight_kg = target_bmi * (1.7 ** 2)
        weight_loss_needed = current_weight_kg - target_weight_kg
        recommendations.append({'category': 'Weight Management', 'priority': 'HIGH', 'impact': f'Potential savings: ${potential_savings:,.0f}/year (₹{potential_savings*USD_TO_INR:,.0f})', 'action': f'Reduce BMI to normal range (18.5-24.9) by losing approximately {weight_loss_needed:.1f} kg. Consult a nutritionist, exercise 150 minutes/week, and maintain a balanced diet.', 'timeframe': '12-18 months for sustainable weight loss'})
    elif user_data['bmi'] > 25:
        recommendations.append({'category': 'Weight Optimization', 'priority': 'MEDIUM', 'impact': 'Prevent future cost increases', 'action': 'Maintain healthy weight through regular exercise (30 min daily) and balanced nutrition to prevent moving into obese category.', 'timeframe': 'Ongoing maintenance'})
    if user_data['age'] < 40 and user_data['smoker'].lower() == 'no' and user_data['bmi'] < 25:
        recommendations.append({'category': 'Preventive Health', 'priority': 'MEDIUM', 'impact': 'Long-term cost stability', 'action': 'Maintain current healthy lifestyle: annual checkups, healthy diet, regular exercise, stress management, and adequate sleep.', 'timeframe': 'Ongoing'})
    similar_profile = database[(database['age'].between(user_data['age']-5, user_data['age']+5)) & (database['bmi'].between(user_data['bmi']-3, user_data['bmi']+3)) & (database['smoker'] == user_data['smoker'].lower())]
    if len(similar_profile) > 0:
        percentile = (similar_profile['predicted_cost'] < prediction_usd).mean() * 100
        if percentile > 75:
            recommendations.append({'category': 'Cost Optimization', 'priority': 'MEDIUM', 'impact': f'You are in the top 25% cost bracket for similar profiles', 'action': 'Compare insurance providers, consider high-deductible health plans with HSA, review coverage needs, and look for employer wellness program discounts.', 'timeframe': 'Next policy renewal'})
    recommendations.append({'category': 'General Wellness', 'priority': 'LOW', 'impact': 'Overall health improvement', 'action': 'Regular health screenings, maintain healthy habits, manage stress, get 7-8 hours sleep, stay hydrated, and build strong social connections.', 'timeframe': 'Ongoing lifestyle'})
    if cost_diff_pct > 50:
        recommendations.append({'category': 'Immediate Action Required', 'priority': 'CRITICAL', 'impact': f'Your cost is {cost_diff_pct:.1f}% higher than average', 'action': 'Focus on high-priority recommendations immediately. Consider consulting with a health coach or financial advisor specialized in healthcare costs.', 'timeframe': 'Start within 1 month'})
    return recommendations
def create_html_report(user_data, prediction_usd, prediction_inr, database, recommendations, fig):
    avg_cost = database['predicted_cost'].mean()
    cost_diff = prediction_usd - avg_cost
    cost_diff_pct = (cost_diff / avg_cost) * 100
    age_group = pd.cut([user_data['age']], bins=[0, 30, 40, 50, 100], labels=['18-30', '31-40', '41-50', '51+'])[0]
    bmi_category = pd.cut([user_data['bmi']], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])[0]
    similar_profile = database[(database['age'].between(user_data['age']-5, user_data['age']+5)) & (database['smoker'] == user_data['smoker'].lower())]
    percentile = (similar_profile['predicted_cost'] < prediction_usd).mean() * 100
    chart_html = fig.to_html(include_plotlyjs='cdn', div_id='chart')
    html_content = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Health Insurance Cost Analysis Report</title><style>body{{font-family:Arial,sans-serif;margin:40px;background-color:#f5f5f5;}}.container{{max-width:1200px;margin:0 auto;background-color:white;padding:40px;box-shadow:0 0 10px rgba(0,0,0,0.1);}}h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px;}}h2{{color:#34495e;margin-top:30px;border-left:4px solid #3498db;padding-left:10px;}}.info-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin:20px 0;}}.info-box{{background-color:#ecf0f1;padding:15px;border-radius:5px;}}.info-label{{font-weight:bold;color:#7f8c8d;}}.info-value{{font-size:1.2em;color:#2c3e50;margin-top:5px;}}.metric-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin:20px 0;}}.metric-box{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:20px;border-radius:10px;text-align:center;}}.metric-label{{font-size:0.9em;opacity:0.9;}}.metric-value{{font-size:1.8em;font-weight:bold;margin:10px 0;}}.metric-sub{{font-size:0.85em;opacity:0.85;}}.recommendation{{background-color:#fff;border-left:4px solid #3498db;padding:15px;margin:15px 0;box-shadow:0 2px 4px rgba(0,0,0,0.1);}}.priority-CRITICAL{{border-left-color:#e74c3c;}}.priority-HIGH{{border-left-color:#e67e22;}}.priority-MEDIUM{{border-left-color:#f39c12;}}.priority-LOW{{border-left-color:#27ae60;}}.rec-header{{font-size:1.1em;font-weight:bold;margin-bottom:10px;}}.rec-detail{{margin:5px 0;padding-left:15px;}}.footer{{margin-top:40px;padding-top:20px;border-top:2px solid #ecf0f1;color:#7f8c8d;font-size:0.9em;}}.timestamp{{text-align:right;color:#95a5a6;font-style:italic;}}</style></head><body><div class="container"><h1>🏥 Health Insurance Cost Analysis Report</h1><div class="timestamp">Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div><h2>Personal Information</h2><div class="info-grid"><div class="info-box"><div class="info-label">Age</div><div class="info-value">{user_data['age']} years</div></div><div class="info-box"><div class="info-label">Gender</div><div class="info-value">{user_data['sex'].capitalize()}</div></div><div class="info-box"><div class="info-label">BMI</div><div class="info-value">{user_data['bmi']:.1f}</div></div><div class="info-box"><div class="info-label">Children</div><div class="info-value">{user_data['children']}</div></div><div class="info-box"><div class="info-label">Smoker</div><div class="info-value">{user_data['smoker'].capitalize()}</div></div><div class="info-box"><div class="info-label">Region</div><div class="info-value">{user_data['region'].capitalize()}</div></div></div><h2>Cost Prediction</h2><div class="metric-grid"><div class="metric-box"><div class="metric-label">Annual Cost</div><div class="metric-value">${prediction_usd:,.2f}</div><div class="metric-sub">₹{prediction_inr:,.2f}</div></div><div class="metric-box"><div class="metric-label">Monthly Cost</div><div class="metric-value">${prediction_usd/12:,.2f}</div><div class="metric-sub">₹{prediction_inr/12:,.2f}</div></div><div class="metric-box"><div class="metric-label">vs Average</div><div class="metric-value">{cost_diff_pct:+.1f}%</div><div class="metric-sub">${cost_diff:+,.2f}</div></div></div><h2>Visual Analysis</h2>{chart_html}<h2>Personalized Recommendations</h2>{''.join([f"<div class='recommendation priority-{rec['priority']}'><div class='rec-header'>{rec['category']} - {rec['priority']} Priority</div><div class='rec-detail'><strong>Impact:</strong> {rec['impact']}</div><div class='rec-detail'><strong>Action:</strong> {rec['action']}</div><div class='rec-detail'><strong>Timeframe:</strong> {rec['timeframe']}</div></div>" for rec in recommendations])}<h2>Statistical Comparison</h2><div class="info-grid"><div class="info-box"><div class="info-label">Age Group</div><div class="info-value">{age_group}</div></div><div class="info-box"><div class="info-label">BMI Category</div><div class="info-value">{bmi_category}</div></div><div class="info-box"><div class="info-label">Cost Percentile</div><div class="info-value">{percentile:.1f}th</div></div><div class="info-box"><div class="info-label">Database Average</div><div class="info-value">${avg_cost:,.2f}</div></div></div><div class="footer"><h2>Disclaimer</h2><p>This report is generated for informational purposes only and should not be considered as medical or financial advice.</p></div></div></body></html>"""
    return BytesIO(html_content.encode('utf-8'))
def create_text_report(user_data, prediction_usd, prediction_inr, database, recommendations):
    avg_cost = database['predicted_cost'].mean()
    cost_diff = prediction_usd - avg_cost
    cost_diff_pct = (cost_diff / avg_cost) * 100
    lines = ["="*80, "HEALTH INSURANCE COST ANALYSIS REPORT", "="*80, f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", "="*80, "PERSONAL INFORMATION", "="*80, f"Age: {user_data['age']}", f"Gender: {user_data['sex'].capitalize()}", f"BMI: {user_data['bmi']:.1f}", f"Number of Children: {user_data['children']}", f"Smoker: {user_data['smoker'].capitalize()}", f"Region: {user_data['region'].capitalize()}", "="*80, "COST PREDICTION", "="*80, f"Annual Cost (USD): ${prediction_usd:,.2f}", f"Annual Cost (INR): ₹{prediction_inr:,.2f}", f"Monthly Cost (USD): ${prediction_usd/12:,.2f}", f"Monthly Cost (INR): ₹{prediction_inr/12:,.2f}", f"Database Average: ${avg_cost:,.2f}", f"Difference from Average: ${cost_diff:,.2f} ({cost_diff_pct:+.1f}%)", "="*80, "PERSONALIZED RECOMMENDATIONS", "="*80]
    for i, rec in enumerate(recommendations, 1):
        lines += [f"{i}. {rec['category']} (Priority: {rec['priority']})", "-"*80, f"Impact: {rec['impact']}", f"Action: {rec['action']}", f"Timeframe: {rec['timeframe']}"]
    lines += ["="*80, "DISCLAIMER", "="*80, "This report is generated for informational purposes only and should not be considered as medical or financial advice.", "="*80]
    return BytesIO("\n".join(lines).encode('utf-8'))
st.markdown("# 🏥 Health Insurance Cost Predictor")
st.markdown("Enter your details to predict your annual medical insurance costs and receive personalized recommendations")
st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Cost", "📊 Model Metrics", "🎯 Predictions & Residuals", "🏆 Best Model Insights"])
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30)
        bmi = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
        children = st.slider("Number of Children", 0, 5, 0)
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
    st.divider()
    if st.button("🔮 Predict Insurance Cost", type="primary", use_container_width=True):
        try:
            sex_encoded = le_sex.transform([sex.lower()])[0]
            smoker_encoded = le_smoker.transform([smoker.lower()])[0]
            region_encoded = le_region.transform([region.lower()])[0]
            input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
            if model is not None:
                prediction_usd = model.predict(input_data)[0]
            else:
                prediction_usd = calculate_cost(age, sex_encoded, bmi, children, smoker_encoded, region_encoded)
            prediction_usd = max(1121.87, min(63770.43, prediction_usd))
            prediction_inr = prediction_usd * USD_TO_INR
            user_data = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
            avg_cost = database['predicted_cost'].mean()
            cost_diff = prediction_usd - avg_cost
            cost_diff_pct = (cost_diff / avg_cost) * 100
            st.success("### 💰 Prediction Results")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Annual Cost (USD)", f"${prediction_usd:,.2f}", delta=f"{cost_diff_pct:+.1f}% vs avg")
                st.caption(f"₹{prediction_inr:,.2f} INR")
            with col_r2:
                st.metric("Monthly Cost (USD)", f"${prediction_usd/12:,.2f}")
                st.caption(f"₹{prediction_inr/12:,.2f} INR")
            with col_r3:
                st.metric("Database Average (USD)", f"${avg_cost:,.2f}")
                st.caption(f"₹{avg_cost*USD_TO_INR:,.2f} INR")
            if smoker == "Yes":
                risk_level, risk_color = "High", "🔴"
            elif bmi > 30:
                risk_level, risk_color = "Medium", "🟡"
            else:
                risk_level, risk_color = "Low", "🟢"
            st.info(f"**Risk Assessment:** {risk_color} {risk_level} Risk")
            st.divider()
            st.markdown("### 📊 Comparative Analysis")
            fig = create_comparison_charts(user_data, prediction_usd, database)
            st.plotly_chart(fig, use_container_width=True)
            st.divider()
            st.markdown("### 💡 Personalized Recommendations")
            recommendations = generate_recommendations(user_data, prediction_usd, database)
            priority_colors = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
            for rec in recommendations:
                with st.expander(f"{priority_colors.get(rec['priority'], '⚪')} {rec['category']} — {rec['priority']} Priority"):
                    st.markdown(f"**Impact:** {rec['impact']}")
                    st.markdown(f"**Recommended Action:** {rec['action']}")
                    st.markdown(f"**Timeframe:** {rec['timeframe']}")
            st.divider()
            st.markdown("### 📥 Download Complete Report")
            html_buffer = create_html_report(user_data, prediction_usd, prediction_inr, database, recommendations, fig)
            txt_buffer = create_text_report(user_data, prediction_usd, prediction_inr, database, recommendations)
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(label="📊 Download HTML Report (with Charts)", data=html_buffer, file_name=f"insurance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html", mime="text/html", use_container_width=True)
            with col_dl2:
                st.download_button(label="📄 Download Text Report", data=txt_buffer, file_name=f"insurance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain", use_container_width=True)
            st.divider()
            with st.expander("📋 Additional Cost Factors"):
                factors = []
                if smoker == "Yes":
                    factors.append("• Smoking significantly increases insurance costs (typically 2-3x higher)")
                if bmi > 30:
                    factors.append("• BMI over 30 may increase premiums by 20-50%")
                if age > 50:
                    factors.append("• Age over 50 typically increases costs due to higher health risks")
                if children > 2:
                    factors.append("• Multiple dependents may affect family plan costs")
                if cost_diff_pct > 25:
                    factors.append(f"• Your cost is {cost_diff_pct:.1f}% above average — review recommendations")
                for factor in factors:
                    st.write(factor)
                if not factors:
                    st.write("✅ No significant risk factors identified — maintain healthy lifestyle!")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.exception(e)
with tab2:
    st.markdown("## 📊 Model Metrics — All 3 Algorithms")
    results = train_and_evaluate_models()
    model_names = list(results.keys())
    palette = {"Linear Regression": "#636EFA", "Random Forest": "#00CC96", "Gradient Boosting": "#EF553B"}
    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1, c2, c3], model_names):
        r = results[name]
        color = palette[name]
        col.markdown(f"""<div style="background:linear-gradient(135deg,{color}22,{color}11);border-left:5px solid {color};border-radius:10px;padding:16px 14px;"><h4 style="margin:0 0 10px 0;color:{color};">{name}</h4><table style="width:100%;font-size:0.9em;"><tr><td>R² Score</td><td align="right"><b>{r['R2']}</b></td></tr><tr><td>MAE</td><td align="right"><b>${r['MAE']:,.2f}</b></td></tr><tr><td>RMSE</td><td align="right"><b>${r['RMSE']:,.2f}</b></td></tr><tr><td>MSE</td><td align="right"><b>${r['MSE']:,.2f}</b></td></tr><tr><td>MAPE</td><td align="right"><b>{r['MAPE']:.2f}%</b></td></tr></table></div>""", unsafe_allow_html=True)
    st.markdown("####")
    rows = [{"Model": n, "R² Score": results[n]["R2"], "MAE ($)": results[n]["MAE"], "RMSE ($)": results[n]["RMSE"], "MSE ($²)": results[n]["MSE"], "MAPE (%)": results[n]["MAPE"]} for n in model_names]
    df_metrics = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df_metrics.style.highlight_max(subset=["R² Score"], color="#d4edda").highlight_min(subset=["MAE ($)", "RMSE ($)", "MSE ($²)", "MAPE (%)"], color="#d4edda"), use_container_width=True)
    left, right = st.columns(2)
    with left:
        fig_r2 = go.Figure()
        for name in model_names:
            fig_r2.add_trace(go.Bar(x=[name], y=[results[name]["R2"]], marker_color=palette[name], text=[f"{results[name]['R2']}"], textposition="outside", width=0.4))
        fig_r2.update_layout(title="R² Score", height=280, showlegend=False, yaxis_range=[0, 1.15], plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10))
        st.plotly_chart(fig_r2, use_container_width=True)
    with right:
        fig_mape = go.Figure()
        for name in model_names:
            fig_mape.add_trace(go.Bar(x=[name], y=[results[name]["MAPE"]], marker_color=palette[name], text=[f"{results[name]['MAPE']:.2f}%"], textposition="outside", width=0.4))
        fig_mape.update_layout(title="MAPE (%)", height=280, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10))
        st.plotly_chart(fig_mape, use_container_width=True)
    left2, right2 = st.columns(2)
    with left2:
        fig_mae = go.Figure()
        for name in model_names:
            fig_mae.add_trace(go.Bar(x=[name], y=[results[name]["MAE"]], marker_color=palette[name], text=[f"${results[name]['MAE']:,.0f}"], textposition="outside", width=0.4))
        fig_mae.update_layout(title="MAE ($)", height=280, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10))
        st.plotly_chart(fig_mae, use_container_width=True)
    with right2:
        fig_rmse = go.Figure()
        for name in model_names:
            fig_rmse.add_trace(go.Bar(x=[name], y=[results[name]["RMSE"]], marker_color=palette[name], text=[f"${results[name]['RMSE']:,.0f}"], textposition="outside", width=0.4))
        fig_rmse.update_layout(title="RMSE ($)", height=280, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10))
        st.plotly_chart(fig_rmse, use_container_width=True)
with tab3:
    st.markdown("## 🎯 Predictions & Residuals")
    results = train_and_evaluate_models()
    model_names = list(results.keys())
    palette = {"Linear Regression": "#636EFA", "Random Forest": "#00CC96", "Gradient Boosting": "#EF553B"}
    fig_avp = make_subplots(rows=1, cols=3, subplot_titles=model_names, shared_yaxes=True)
    for i, name in enumerate(model_names, 1):
        y_test = results[name]["y_test"]
        y_pred = results[name]["y_pred"]
        sample = np.random.choice(len(y_test), min(400, len(y_test)), replace=False)
        fig_avp.add_trace(go.Scatter(x=y_test[sample], y=y_pred[sample], mode="markers", marker=dict(color=palette[name], size=4, opacity=0.55), showlegend=False), row=1, col=i)
        lo, hi = y_test.min(), y_test.max()
        fig_avp.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="black", dash="dash", width=1.5), showlegend=False), row=1, col=i)
    fig_avp.update_layout(height=320, title="Actual vs Predicted", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20))
    fig_avp.update_xaxes(title_text="Actual ($)")
    fig_avp.update_yaxes(title_text="Predicted ($)", col=1)
    st.plotly_chart(fig_avp, use_container_width=True)
    fig_res = make_subplots(rows=1, cols=3, subplot_titles=model_names, shared_yaxes=True)
    for i, name in enumerate(model_names, 1):
        y_test = results[name]["y_test"]
        y_pred = results[name]["y_pred"]
        resids = y_test - y_pred
        sample = np.random.choice(len(resids), min(400, len(resids)), replace=False)
        fig_res.add_trace(go.Scatter(x=y_pred[sample], y=resids[sample], mode="markers", marker=dict(color=palette[name], size=4, opacity=0.5), showlegend=False), row=1, col=i)
        fig_res.add_hline(y=0, line_dash="dash", line_color="black", line_width=1.5, row=1, col=i)
    fig_res.update_layout(height=320, title="Residuals (Actual − Predicted vs Predicted)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=20))
    fig_res.update_xaxes(title_text="Predicted ($)")
    fig_res.update_yaxes(title_text="Residual ($)", col=1)
    st.plotly_chart(fig_res, use_container_width=True)
    fig_hist = go.Figure()
    for name in model_names:
        abs_err = np.abs(results[name]["y_test"] - results[name]["y_pred"])
        fig_hist.add_trace(go.Histogram(x=abs_err, name=name, marker_color=palette[name], opacity=0.65, nbinsx=40))
    fig_hist.update_layout(barmode="overlay", title="Absolute Error Distribution", xaxis_title="Absolute Error ($)", yaxis_title="Count", height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend=dict(x=0.75, y=0.95), margin=dict(t=50, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)
with tab4:
    st.markdown("## 🏆 Best Model Insights")
    results = train_and_evaluate_models()
    model_names = list(results.keys())
    palette = {"Linear Regression": "#636EFA", "Random Forest": "#00CC96", "Gradient Boosting": "#EF553B"}
    best_name = "Gradient Boosting"
    st.success(f"**Best Model: {best_name}** — R² = {results[best_name]['R2']} | MAE = ${results[best_name]['MAE']:,.2f} | RMSE = ${results[best_name]['RMSE']:,.2f}")
    left, right = st.columns(2)
    with left:
        metrics_for_radar = ["R2", "MAE", "RMSE", "MAPE"]
        radar_labels = ["R² Score", "MAE", "RMSE", "MAPE"]
        raw = {m: np.array([results[n][m] for n in model_names], dtype=float) for m in metrics_for_radar}
        norm = {}
        for m in metrics_for_radar:
            vals = raw[m]
            mn, mx = vals.min(), vals.max()
            norm[m] = (vals - mn) / (mx - mn + 1e-9) if m == "R2" else 1 - (vals - mn) / (mx - mn + 1e-9)
        fig_radar = go.Figure()
        categories = radar_labels + [radar_labels[0]]
        for i, name in enumerate(model_names):
            vals = [norm[m][i] for m in metrics_for_radar] + [norm[metrics_for_radar[0]][i]]
            fig_radar.add_trace(go.Scatterpolar(r=vals, theta=categories, fill="toself", name=name, line_color=palette[name], fillcolor=palette[name], opacity=0.35))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Performance Radar (higher = better)", height=340, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=10))
        st.plotly_chart(fig_radar, use_container_width=True)
    with right:
        best_model = results[best_name]["model"]
        feature_names = ['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region']
        if hasattr(best_model, "feature_importances_"):
            importance = best_model.feature_importances_
            label = "Feature Importance"
        elif hasattr(best_model, "coef_"):
            importance = np.abs(best_model.coef_)
            label = "Absolute Coefficient"
        else:
            importance = None
            label = ""
        if importance is not None:
            df_imp = pd.DataFrame({"Feature": feature_names, label: importance}).sort_values(label, ascending=True)
            fig_imp = px.bar(df_imp, x=label, y="Feature", orientation="h", title=f"{label} — {best_name}", color=label, color_continuous_scale="Blues")
            fig_imp.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=50, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)
    summary_rows = [{"Model": n, "R² Score": results[n]["R2"], "MAE ($)": results[n]["MAE"], "RMSE ($)": results[n]["RMSE"], "MSE ($²)": results[n]["MSE"], "MAPE (%)": results[n]["MAPE"], "Best?": "✅" if n == best_name else ""} for n in model_names]
    st.dataframe(pd.DataFrame(summary_rows).set_index("Model"), use_container_width=True)
