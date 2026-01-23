from flask import render_template,Flask

app=Flask(__name__,template_folder="templates")


import plotly.express as px
@app.route("/")
def plot():
    x = [1, 2, 3, 4, 5]
    y = [10, 15, 13, 17, 20]

    fig = px.line(
        x=x,
        y=y,
        markers=True,
        title="Plotly Pan + Hover Values Example"
    )

    # Enable pan by default
    fig.update_layout(
        dragmode="pan",
        hovermode="closest"
    )

    # Show values when hovering points
    fig.update_traces(
        hovertemplate="<br> X: %{x}<br>Y: %{y}<extra></extra>"
    )

    return fig.to_html(full_html=False)





def notnow():
    import pandas as pd

    # create empty DataFrame with only column definitions
    df = pd.DataFrame(columns=["narrative", "product"])

    # save to path
    file_path = "./data/misclassified_data.csv"
    df.to_csv(file_path, index=False)

    print("CSV created at:", file_path)



if __name__=="__main__":
    app.run(debug=True)