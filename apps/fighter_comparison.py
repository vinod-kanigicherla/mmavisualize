import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd

from datetime import timedelta, datetime

from app import app
from data_processing.data_info import FIGHTERINFO, FIGHTER_DROPDOWN_OPTIONS, STATS_DICT, STATS_DICT_REVERSE, DATES, get_min_date, get_cmp_df 


def get_daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)


TB_STYLE = {"margin-top": "1rem", "margin-bottom": "2rem"}
TBC_STYLE = {"margin-top": "2rem", "margin-bottom": "2rem", "text-align": "center"}

def generate_table(df, id):
    return dbc.Table.from_dataframe(df, bordered=True, hover=True, style=TBC_STYLE, id=id)

def generate_bold_p(text):
    return html.P(text, style={"font-weight": "bold", "text-align": "center", "margin-bottom": "0.3rem"})


fighter_dropdowns = [
    dcc.Dropdown(
        id="fighter-cmp-dropdown-1", 
        options=FIGHTER_DROPDOWN_OPTIONS,
        placeholder="Select Fighter 1"
    ), 
    dcc.Dropdown(
        id="fighter-cmp-dropdown-2", 
        options=FIGHTER_DROPDOWN_OPTIONS,
        placeholder="Select Fighter 2"
    ) 
    ] 
    
fighter_dropdowns_row = html.Div(
    [
        dbc.Row([
            dbc.Col(html.Div([generate_bold_p("Fighter 1"), fighter_dropdowns[0]]), width=6), 
            dbc.Col(html.Div([generate_bold_p("Fighter 2"), fighter_dropdowns[1]]), width=6)])
    ], 
    style=TB_STYLE
)


cmp_stats_dropdown = dcc.Dropdown(
    id="cmp-stats-dropdown", 
    options=[{"label": sn, "value": STATS_DICT[sn]} for sn in STATS_DICT.keys()], 
    value=["wins", "losses", "curr_win_strk", "sig_str_acc_pct"], 
    multi=True
) 

cmp_stats_dropdown_row = html.Div([generate_bold_p("Choose Comparison Stats"), cmp_stats_dropdown], style=TB_STYLE) 

date_pickers = [
    dcc.DatePickerSingle(
        id="cmp-date-picker-1",
        min_date_allowed=DATES[0],
        max_date_allowed=datetime.now(), 
        initial_visible_month=datetime.now(),
        date=datetime.now()
    ), 
    dcc.DatePickerSingle(
        id="cmp-date-picker-2",
        min_date_allowed=DATES[0],
        max_date_allowed=datetime.now(), 
        initial_visible_month=datetime.now(), 
        date=datetime.now()
    )
]

date_pickers_row = html.Div(
    [
        dbc.Row([
            dbc.Col(html.Div([generate_bold_p("Fighter 1 Date"), date_pickers[0]], style={"text-align": "center"}), width=6), 
            dbc.Col(html.Div([generate_bold_p("Fighter 2 Date"), date_pickers[1]], style={"text-align": "center"}), width=6)])
    ], 
    style=TB_STYLE
)

table_row = html.Div(id="cmp-stats-table-row", style=TB_STYLE)

cmp_stat_history_dropdown_row = html.Div(id="cmp-stat-history-dropdown-row", style=TB_STYLE)

cmp_stat_history_graph_row = html.Div(id="cmp-stat-history-graph-row", style=TB_STYLE)


layout = dbc.Container([fighter_dropdowns_row, date_pickers_row, cmp_stats_dropdown_row, table_row, cmp_stat_history_dropdown_row, cmp_stat_history_graph_row])


@app.callback(Output("cmp-date-picker-1", "min_date_allowed"), [Input("fighter-cmp-dropdown-1", "value")])
def update_date_picker_1(f1):
    if not f1:
        return  
    f1_min_date = get_min_date(f1)
    return f1_min_date 

@app.callback(Output("cmp-date-picker-2", "min_date_allowed"), [Input("fighter-cmp-dropdown-2", "value")])
def update_date_picker_2(f2):
    if not f2:
        return  
    f2_min_date = get_min_date(f2)
    return f2_min_date 

@app.callback(Output("cmp-stats-table-row", "children"), 
    [Input("fighter-cmp-dropdown-1", "value"), Input("fighter-cmp-dropdown-2", "value"), 
    Input("cmp-date-picker-1", "date"), Input("cmp-date-picker-2", "date")], Input("cmp-stats-dropdown", "value"))
def generate_cmp_table(f1, f2, date1, date2, stats):
    if not (f1 and f2 and date1 and date2 and stats):
        return None
    return generate_table(get_cmp_df(f1, f2, date1, date2, stats), "cmp-stats-table") 


@app.callback(
    Output('cmp-stat-history-dropdown-row', 'children'),
    [Input("fighter-cmp-dropdown-1", "value"), Input("fighter-cmp-dropdown-2", "value"), Input('cmp-stats-dropdown', 'value')])
def get_history_comparison_dropdown(f1, f2, cmp_stats):
    if f1 and f2:
        heading = generate_bold_p("Choose Stat for Historical Visualization:")
        dropdown = dcc.Dropdown(
                id="cmp-stat-history-dropdown", 
                options=[{"label": STATS_DICT_REVERSE[s], "value": s} for s in cmp_stats]) 
        
        return [heading, dropdown]


@app.callback(
    Output('cmp-stat-history-graph-row', 'children'),
    [Input("fighter-cmp-dropdown-1", "value"), Input("fighter-cmp-dropdown-2", "value"), 
    Input("cmp-date-picker-1", "min_date_allowed"), Input("cmp-date-picker-2", "min_date_allowed")], 
    Input('cmp-stat-history-dropdown', 'value'))
def get_history_comparison_graph(f1, f2, d1, d2, stat):
    graph=None 
    if f1 and f2 and stat:
        min_date = d1 if d1<d2 else d2 
        print(f"d1: {d1}, d2: {d2}, min date: {min_date}, type: {type(min_date)}")
        dates = DATES[DATES.to_series().apply(lambda x: x >= pd.to_datetime(min_date))]
        x = [str(d) for d in dates] 
        y_f1 = list(FIGHTERINFO.loc[dates, f1][stat].values)
        y_f2 = list(FIGHTERINFO.loc[dates, f2][stat].values)
        graph = dcc.Graph(
                id="cmp-stat-history-graph", 
                figure={
                    'data': [
                        {'x': x, 'y': y_f1, 'type': 'line', 'name': f1}, 
                        {'x': x, 'y': y_f2, 'type': 'line', 'name': f2},
                    ], 
                    'layout': {
                        'title': f"Historical Comparison of {STATS_DICT_REVERSE[stat]}: {f1} vs. {f2}"
                    }
                } 
            )
    
    return [graph] 





    