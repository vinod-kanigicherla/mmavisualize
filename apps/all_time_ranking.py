import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd

from datetime import timedelta, datetime

from app import app
from data_processing.data_info import STATS_DICT, DATES, get_at_ranking_df 


TB_STYLE = {"margin-top": "1rem", "margin-bottom": "2rem"}
TBC_STYLE = {"margin-top": "2rem", "margin-bottom": "2rem", "text-align": "center"}

def generate_table(df, id):
    return dbc.Table.from_dataframe(df, bordered=True, hover=True, style=TBC_STYLE, id=id)

def generate_bold_p(text):
    return html.P(text, style={"font-weight": "bold", "text-align": "center", "margin-bottom": "0.3rem"})

stats_dropdown = dcc.Dropdown(
    id="at-stats-dropdown", 
    options=[{"label": sn, "value": STATS_DICT[sn]} for sn in STATS_DICT.keys()]
) 

stats_dropdown_row = html.Div([generate_bold_p("Choose Stat for Ranking:"), stats_dropdown], style=TBC_STYLE) 

date_picker = dcc.DatePickerSingle(
        id="at-date-picker",
        min_date_allowed=DATES[0],
        max_date_allowed=datetime.now(), 
        initial_visible_month=datetime.now(),
        date=datetime.now()
    )

date_picker_row = html.Div([generate_bold_p("Pick Date of Stats:"), date_picker], style=TBC_STYLE)

table_row = html.Div(id="at-ranking-table-row", style=TB_STYLE)

layout = dbc.Container([stats_dropdown_row, date_picker_row, table_row])

@app.callback(Output("at-ranking-table-row", "children"), [Input("at-stats-dropdown", "value"), Input("at-date-picker", "date")])
def generate_ranking_table(stat, date):
    print("at ranking table")
    return generate_table(get_at_ranking_df(stat, date), "at-ranking-table") 



