import streamlit as st
from bokeh.plotting import figure


if "p" not in st.session_state:
    st.session_state["p"] = 0.5

st.subheader("Bernoulli Distribution")

p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, key="p")
x = [0, 1]
y = [1 - p, p]

fig = figure(title='Bernoulli Distribution', x_axis_label='Outcome', y_axis_label='Probability')
fig.vbar(x=x, top=y, width=0.5)
st.bokeh_chart(fig, use_container_width=True)