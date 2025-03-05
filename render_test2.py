


import dash
from dash import Dash, dcc, html, Input, Output
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from statistics import mean
import glob
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter
#import dash_core_components as dcc
#import dash_html_components as html
import plotly.graph_objects as go
import math
import gdown

# Daten einlesen
#dateipfad = '/content/drive/MyDrive/DIIHK/Data/gesamter_Datensatz_nach_Land_sortiert.csv'
#all_sorted = pd.read_csv(dateipfad, encoding='utf-8')

# https://drive.google.com/file/d/1-Qn9Zg4sxmAwhxsATyJNQ--EQZdqAGS8/view?usp=sharing
#url = "https://drive.google.com/uc?id=1-Qn9Zg4sxmAwhxsATyJNQ--EQZdqAGS8"
#all_sorted = pd.read_csv(url, encoding='utf-8')



# Sicherstellen, dass die Zeitraum-Spalte als Datum erkannt wird
#all_sorted['Zeitraum'] = pd.to_datetime(all_sorted['Zeitraum'])
#all_sorted['Jahr'] = all_sorted['Zeitraum'].dt.year

# Verwende die ID der Datei aus der Google Drive URL
file_id = '1-Qn9Zg4sxmAwhxsATyJNQ--EQZdqAGS8'
url = f"https://drive.google.com/uc?id={file_id}"

# Die Datei herunterladen
gdown.download(url, 'gesamter_Datensatz_nach_Land_sortiert.csv', quiet=False)

# Danach die CSV-Datei in Pandas laden
all_sorted = pd.read_csv('gesamter_Datensatz_nach_Land_sortiert.csv', encoding='utf-8')

# Sicherstellen, dass die Zeitraum-Spalte als Datum erkannt wird
all_sorted['Zeitraum'] = pd.to_datetime(all_sorted['Zeitraum'])
all_sorted['Jahr'] = all_sorted['Zeitraum'].dt.year


### Each year's export, import and trade volume for each country + wachstum + differenz + ranking

# Gruppieren nach Land und Jahr
all_sorted['Jahr'] = all_sorted['Zeitraum'].dt.year
all_sorted['Monat'] = all_sorted['Zeitraum'].dt.month

# Berechnungen für Export, Import und Handelsvolumen
df_grouped = all_sorted.groupby(['Land', 'Jahr']).agg(
    export_wert=('Ausfuhr: Wert', 'sum'),
    import_wert=('Einfuhr: Wert', 'sum')
).reset_index()

# Berechnen des Handelsvolumenwerts
df_grouped['handelsvolumen_wert'] = df_grouped['export_wert'] + df_grouped['import_wert']

# Berechnen der Handelsbilanz
df_grouped['handelsbilanz'] = df_grouped['export_wert'] - df_grouped['import_wert']

# Definieren ob Überschuss oder Defizit
df_grouped['handelsbilanz_status'] = df_grouped['handelsbilanz'].apply(lambda x: 'Überschuss' if x > 0 else 'Defizit')

# Ranking für jedes Jahr nach den 3 Kategorien (Export, Import, Handelsvolumen)
df_grouped['export_ranking'] = df_grouped.groupby('Jahr')['export_wert'].rank(ascending=False)
df_grouped['import_ranking'] = df_grouped.groupby('Jahr')['import_wert'].rank(ascending=False)
df_grouped['handelsvolumen_ranking'] = df_grouped.groupby('Jahr')['handelsvolumen_wert'].rank(ascending=False)

# Top 10 Länder nach Jahr und Kategorie
top_10_export = df_grouped[df_grouped['export_ranking'] <= 10]
top_10_import = df_grouped[df_grouped['import_ranking'] <= 10]
top_10_handelsvolumen = df_grouped[df_grouped['handelsvolumen_ranking'] <= 10]

# Berechnen des Wachstums für Export, Import und Handelsvolumen
df_grouped['export_wachstum'] = df_grouped.groupby('Land')['export_wert'].pct_change() * 100
df_grouped['import_wachstum'] = df_grouped.groupby('Land')['import_wert'].pct_change() * 100
df_grouped['handelsvolumen_wachstum'] = df_grouped.groupby('Land')['handelsvolumen_wert'].pct_change() * 100

# Für 2008 setzen wir das Wachstum auf 0, da es kein Vorjahr gibt
df_grouped.loc[df_grouped['Jahr'] == 2008, 'export_wachstum'] = 0
df_grouped.loc[df_grouped['Jahr'] == 2008, 'import_wachstum'] = 0
df_grouped.loc[df_grouped['Jahr'] == 2008, 'handelsvolumen_wachstum'] = 0

# Berechnen des Rankings für das Wachstum in jeder Kategorie
df_grouped['export_wachstum_ranking'] = df_grouped.groupby('Jahr')['export_wachstum'].rank(ascending=False)
df_grouped['import_wachstum_ranking'] = df_grouped.groupby('Jahr')['import_wachstum'].rank(ascending=False)
df_grouped['handelsvolumen_wachstum_ranking'] = df_grouped.groupby('Jahr')['handelsvolumen_wachstum'].rank(ascending=False)

# Berechnung der absoluten Differenzen zum Vorjahr
df_grouped['export_differenz'] = df_grouped.groupby('Land')['export_wert'].diff()
df_grouped['import_differenz'] = df_grouped.groupby('Land')['import_wert'].diff()
df_grouped['handelsvolumen_differenz'] = df_grouped.groupby('Land')['handelsvolumen_wert'].diff()

# Für 2008 setzen wir die Differenzen auf 0, da es kein Vorjahr gibt
df_grouped.loc[df_grouped['Jahr'] == 2008, 'export_differenz'] = 0
df_grouped.loc[df_grouped['Jahr'] == 2008, 'import_differenz'] = 0
df_grouped.loc[df_grouped['Jahr'] == 2008, 'handelsvolumen_differenz'] = 0

# Ausgabe des DataFrames mit den neuen Spalten
df_grouped

#!pip freeze > requirements.txt
#from google.colab import files
#files.download('requirements.txt')



# Funktion zur Formatierung der Y-Achse
def formatter(value):
    if value >= 1e9:
        return f'{value / 1e9:.0f} Mrd'
    elif value >= 1e6:
        return f'{value / 1e6:.0f} Mio'
    elif value >= 1e3:
        return f'{value / 1e3:.0f} K'
    else:
        return str(value)


# Funktion zur Formatierung der Y-Achse
# def formatter(value):
#     if value >= 1e9:
#         return f'{value / 1e9:.1f} Mrd'
#     elif value >= 1e6:
#         return f'{value / 1e6:.1f} Mio'
#     elif value >= 1e3:
#         return f'{value / 1e3:.1f} K'
#     else:
#         return str(value)

# Gruppierung der Daten nach Jahr
gesamt_deutschland = all_sorted.groupby('Jahr').agg(
    gesamt_export=('Ausfuhr: Wert', 'sum'),
    gesamt_import=('Einfuhr: Wert', 'sum')
).reset_index()

gesamt_deutschland['gesamt_handelsvolumen'] = gesamt_deutschland['gesamt_export'] + gesamt_deutschland['gesamt_import']

# Dash-App erstellen
app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.H1("Deutschlands Handelsentwicklung"),
    dcc.Graph(id='handel_graph'),
])

@app.callback(
    Output('handel_graph', 'figure'),
    Input('handel_graph', 'id')
)
def update_graph(_):
    fig = go.Figure()

    # Linien für Export, Import und Handelsvolumen
    for col, name, color in zip(
        ['gesamt_export', 'gesamt_import', 'gesamt_handelsvolumen'],
        ['Exportvolumen', 'Importvolumen', 'Gesamthandelsvolumen'],
        ['#1f77b4', '#ff7f0e', '#2ca02c']
    ):
        fig.add_trace(go.Scatter(
            x=gesamt_deutschland['Jahr'],
            y=gesamt_deutschland[col],
            mode='lines+markers',
            name=name,
            line=dict(width=2, color=color),
            hovertemplate=f'<b>{name}</b><br>Jahr: %{{x}}<br>Wert: %{{y:,.0f}} €'
        ))

    # Berechnung der maximalen Y-Achse, um die Schrittgröße zu bestimmen
    max_value = max(gesamt_deutschland['gesamt_export'].max(),
                    gesamt_deutschland['gesamt_import'].max(),
                    gesamt_deutschland['gesamt_handelsvolumen'].max())

    # Berechnung der Tick-Schritte basierend auf dem maximalen Wert
    tick_step = 500e9  # 500 Mrd als Standard-Schrittgröße
    tickvals = np.arange(0, max_value + tick_step, tick_step)

    # Layout-Anpassungen
    fig.update_layout(
        title='Entwicklung von Export, Import und Handelsvolumen',
        xaxis_title='Jahr',
        yaxis_title='Wert in €',
        yaxis=dict(
            tickformat=',',
            tickvals=tickvals,
            ticktext=[formatter(val) for val in tickvals]
        ),
        legend=dict(title='Kategorie', bgcolor='rgba(255,255,255,0.7)')
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
