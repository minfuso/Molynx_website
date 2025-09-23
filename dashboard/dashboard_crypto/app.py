import os
import json
import time
from datetime import datetime
import requests
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import tensorflow as tf
import yaml
import joblib
import numpy as np
from zoneinfo import ZoneInfo

from Informer.model.informer import Informer
from Informer.data.normalization import transform_with_scalers
from Informer.data.windowing import df_list_to_sequences


# --- Setting the good path ---

BASE_DIR = os.path.dirname(__file__)  # répertoire courant = dashboard/dashboard_crypto
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- Initialisation du dashboard ---

def init_dashboard(server):
    dash_app = dash.Dash(__name__, 
                    suppress_callback_exceptions=True,
                    server=server,
                    url_base_pathname="/dashboard/dashboard_crypto/",
                    )
    
    
    dash_app.layout = html.Div([
        html.H1("Dashboard Crypto (Binance API)"),

        dcc.Tabs(id="tabs", value="tab1", className="tabs", children=[
            dcc.Tab(label="Prix", value="tab1", className="custom-tab", selected_className="custom-tab--selected"),
            dcc.Tab(label="Bitcoin IA", value="tab2", className="custom-tab", selected_className="custom-tab--selected"),
        ]),

        html.Div(id="tabs-content", className="tabs-content")
    ])
    
    
    @dash_app.callback(
        Output("tabs-content", "children"),
        Input("tabs", "value")
    )
    def render_tab(tab):
        if tab == "tab1":
            return html.Div([
                
                dcc.Dropdown(
                    id="dropdown",
                    options=[{"label": c.capitalize(), "value": c} for c in symbols.keys()],
                    value="bitcoin",
                    className="dropdown",
                ),
                
                html.H2(id="crypto-graph-title"),

                dcc.Graph(id="crypto-graph", className="graph"),
                
                html.H2("Informations supplémentaires"),
                
                dcc.Graph(id="secondary-graph", className="graph"),
                
            ])
        elif tab == "tab2":
            return html.Div([
                
        
                html.Div(
                [
                        html.Div([
                            html.H3("Prédiction IA", className="card-title"),
                            html.P("Prédiction tendance", id="predict-tendency"),
                            html.P("Prédiction stratégie", id="predict-strategy"),
                            html.P("Cliquez sur le bouton pour faire une prédiction", id="predict-date")
                        ], className="card")  # ta classe existante
                    ],
                    className="cards-container"  # un conteneur si tu veux mettre plusieurs cards
                ),
                
                html.Div([
                    html.Button("Rafraîchir la prédiction", id="predict-button", n_clicks=0, className="btn btn-centre"),
                ], className="btn-container"),
                
                
                html.H2("Backtesting"),
                
                html.H3("Simulation d'un portfolio", className="h3_global"),
                
                dcc.Graph(className="graph", figure=fig_backtest),
                            
            ])
    
    
    @dash_app.callback(
        [Output("crypto-graph", "figure"),
        Output("crypto-graph-title", "children")],
        Input("dropdown", "value")
    )
    def update_graph(crypto):
        symbol = symbols[crypto]
        url = "https://api.binance.com/api/v3/klines"
        df = get_data_with_cache(
            f"cache_{symbol}_history.csv", url,
            params={"symbol": symbol, "interval": "1h", "limit": 500},
            max_age=1800
        )

        # df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        
        # Calcule du max de volume
        max_vol = df["volume"].max() if not df["volume"].empty else 0

        # On veut que la borne max de y2 soit telle que le max_vol soit à ~60% de cette borne
        # Donc : upper_bound = max_vol / 0.6
        # Tu peux ajuster le facteur (ici 0.6)
        if max_vol > 0:
            upper_y2 = max_vol / 0.6
        else:
            upper_y2 = None  # ou un fallback

        # Crée une figure avec axe secondaire
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Ajoute la courbe de prix sur l’axe y principal
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["close"],
                name="Prix ($)",
                showlegend=True,
                line=dict(color=colors["heading_color"], width=3),
                # line=dict(color=neon_color, width=2),
                marker=dict(size=0)  # pas de marker si tu veux seulement la ligne
            ),
            secondary_y=False
        )
        
        # Ajoute l’histogramme de volume sur l’axe secondaire Y
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                name="Volume",
                marker_color=colors["header_color_1"],  # couleur semi-transparente
                opacity=0.6
            ),
            secondary_y=True
        )
        
        # Configuration des axes
        y2_axis_kwargs = dict(
            title="Volume",
            color=colors["text_color"],
            showgrid=False
        )
        if upper_y2 is not None:
            y2_axis_kwargs["range"] = [0, upper_y2]
            y2_axis_kwargs["autorange"] = False 
            

        fig.update_layout(
            plot_bgcolor=colors["background_color"],
            paper_bgcolor=colors["background_color"],
            font=dict(color=colors["text_color"], family="Inter"),
            xaxis=dict(
                title="",
                color=colors["text_color"],
                gridcolor=colors["dark_grey"],
                tickangle=45,
            ),
            yaxis=dict(
                title="Prix ($)",
                color=colors["text_color"],
                gridcolor=colors["dark_grey"]
            ),
            yaxis2=y2_axis_kwargs,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )

        title = f"Évolution du prix sur les derniers jours ({crypto.capitalize()})"
        return fig, title
    
    
    @dash_app.callback(
        Output("secondary-graph", "figure"),
        Input("dropdown", "value")
    )
    def update_secondary_graph(crypto):
        symbol = symbols[crypto]
        url = "https://api.binance.com/api/v3/klines"
        df = get_data_with_cache(
            f"cache_{symbol}_history.csv", url,
            params={"symbol": symbol, "interval": "1h", "limit": 500},
            max_age=1800
        )
        
        # Trouver min/max de chaque axe
        y1_min, y1_max = df["MACD"].min(), df["MACD"].max()
        y2_min, y2_max = df["volume_rel20"].min(), df["volume_rel20"].max()
        
        # Assurer que zéro est inclus dans les axes
        y1_min = min(y1_min, 0)
        y2_min = min(y2_min, 0)
        
        # Crée une figure avec axe secondaire
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df["MACD"],   
            mode="lines",          
            name="MACD", 
            showlegend=True,
            line=dict(color=colors["heading_color"], width=3),
            ),
            secondary_y=False
            )
        
        fig.add_trace(go.Bar(
                x=df.index, 
                y=df["volume_rel20"],             
                name="Volume Relatif 20h", 
                showlegend=True,
                marker_color=colors["header_color_1"],
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            plot_bgcolor=colors["background_color"],
            paper_bgcolor=colors["background_color"],
            font=dict(color=colors["text_color"], family="Inter"),
            xaxis=dict(
                title="",
                color=colors["text_color"],
                gridcolor=colors["dark_grey"],
                tickangle=45,
            ),
            yaxis=dict(
                title="",
                color=colors["text_color"],
                gridcolor=colors["dark_grey"],
            ),
            yaxis2=dict(
                showgrid=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Mise à jour des axes pour inclure zéro sur les deux
        fig.update_yaxes(range=[0, y1_max], secondary_y=False, title_text="MACD")
        fig.update_yaxes(range=[0, y2_max], secondary_y=True, title_text="Volume Relatif 20h")

        return fig
    
    
    @dash_app.callback(
        [Output("predict-date", "children"),
        Output("predict-tendency", "children"),
        Output("predict-strategy", "children")],
        Input("predict-button", "n_clicks")
    )
    def update_prediction(n_clicks):
        symbol = symbols["bitcoin"]
        url = "https://api.binance.com/api/v3/klines"
        df = get_data_with_cache(
            f"cache_{symbol}_history.csv", url,
            params={"symbol": symbol, "interval": "1h", "limit": 500},
            max_age=1800
        )
        
        now = datetime.now(ZoneInfo("Europe/Paris"))
        # Par exemple formater comme ça :
        date_string = now.strftime("%d/%m %H:%M:%S")
        date_message = "Mise à jour : " + date_string
        # Prepare data
        X_pred = prepare_data_for_IA(df, features, feature_config, seq_len, scaler, target)
        pred = float(model.predict(np.expand_dims(X_pred[-1], axis=0), verbose=0).squeeze())
        # Treating prediction
        pred_up = pred
        pred_down = 1.0 - pred_up
        
        # Define tendency
        if pred_up >= pred_down:
            tendency = html.Span([
                "Prédiction d’",
                html.Strong("augmentation"),
                " à ",
                html.Strong(f"{pred_up*100:2.1f} %")
            ])
        else :
            tendency = html.Span([
                "Prédiction de ",
                html.Strong("reduction"),
                " à ",
                html.Strong(f"{pred_down*100:2.1f} %")
            ])
            
        # Define strategy
        if pred_up >= 0.54:
            strategy = html.Span([
                "Nous vous conseillons d'",
                html.Strong("acheter"),
                " pour la prochaine heure"
            ]) 
        elif (pred_up > 0.46) and ( pred_up < 0.54):
            strategy = html.Span([
                "Nous vous conseillons de ",
                html.Strong("hold"),
                " pour la prochaine heure"
            ]) 
        else:
            strategy = html.Span([
                "Nous vous conseillons de ",
                html.Strong("vendre"),
                " pour la prochaine heure"
            ]) 
        
        
        return date_message, tendency, strategy
    
    return dash_app

# --- Importation du modèle IA ---
def load_model(model_path:str) -> Informer:
    model = tf.keras.models.load_model(model_path, custom_objects={"Informer": Informer})
    return model

def load_parameters(config_path, scaler_path):
    # Config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    feature_config = cfg["data"]["features_map"]
    features = list(feature_config.keys())
    target=cfg["data"]["target"]
    seq_len = cfg["data"]["sequence_length"]
    # Scalers
    scaler = joblib.load(scaler_path)
    
    return feature_config, features, target, seq_len, scaler   

# --- Prepare data for IA model ---
def prepare_data_for_IA(df, features, feature_config, seq_len, scaler, target):
    df_scaled = df.copy()
    df_scaled = transform_with_scalers([df_scaled], feature_config, scaler)[0]
    X_pred, _ = df_list_to_sequences([df_scaled],  features, target, seq_len)
    return X_pred

# --- Prediction function ---
def make_prediction(X_pred):
    pred = float(model.predict(np.expand_dims(X_pred[-1], axis=0), verbose=0).squeeze())
    return pred

# --- Initialization of the IA ---

model = load_model(os.path.join(DATA_DIR, "informer_best.model.keras"))

config_path = os.path.join(DATA_DIR, "data_config.yaml")
scaler_path = os.path.join(DATA_DIR, "btc_seq24_scalers.pkl")

feature_config, features, target, seq_len, scaler = load_parameters(config_path=config_path, scaler_path=scaler_path)


# --- Fonction de cache ---
def get_data_with_cache(cache_file, url, params=None, max_age=1800, key_check=None):
    if os.path.exists(cache_file):
        if time.time() - os.path.getmtime(cache_file) < max_age:
            try:
                return pd.read_csv(os.path.join(DATA_DIR, cache_file), index_col=0)
            except Exception:
                pass

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data , columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df = prepare_data(df)
    else:
        raise ValueError(f"Format JSON inattendu : {type(data)}")
    
    df.to_csv(os.path.join(DATA_DIR, cache_file), index=True)
    return df

def prepare_data(df):
    # Settings the time and index
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", errors="coerce")
    df = df.set_index("open_time").sort_index()
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_range)
    df.index.name = "open_time"
    
    # Setting other colums
    for column in df.columns:
        if column in ["close_time", "number_of_trades"]:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    
    # Computing the features
    df = compute_features(df)
    
    # Cleaning the dataset
    df = clean_dataset(df)
    
    # Redifine close to open
    df = reassign_close_to_open(df)
    
    # Defining the target
    df = define_future_evolution(df)
    
    
    return df

def load_css_vars(path):
    vars_dict = {}
    with open(path, "r") as f:
        css = f.read()
    matches = re.findall(r'--([\w_-]+):\s*(#[0-9A-Fa-f]{6});', css)
    for name, value in matches:
        vars_dict[name] = value
    return vars_dict

# --- Symbols ---

symbols = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "dogecoin": "DOGEUSDT",
    "cardano": "ADAUSDT",
    "solana": "SOLUSDT"
}

colors = load_css_vars(os.path.join(ASSETS_DIR, "base.css"))

##### Fonction backtesting

def create_comparison_plot(df_bt):
    
    df_bt = df_bt.set_index("open_time").sort_index()
    
    # df_bt doit contenir les colonnes "portfolio", "portfolio_bh", "portfolio_random"
    fig = go.Figure()

    # Trace Strategy (model)
    fig.add_trace(go.Scatter(
        x=df_bt.index,
        y=df_bt["portfolio"],
        mode="lines",
        name="Notre modèle",
        line=dict(color=colors["heading_color"], width=2)
    ))

    # Trace Buy & Hold
    fig.add_trace(go.Scatter(
        x=df_bt.index,
        y=df_bt["portfolio_bh"],
        mode="lines",
        name="Buy & Hold",
        line=dict(color="white", width=2)
    ))

    # Trace Random
    fig.add_trace(go.Scatter(
        x=df_bt.index,
        y=df_bt["portfolio_random"],
        mode="lines",
        name="Hasard",
        line=dict(color="grey", width=2),
        opacity=0.7
    ))

    # Mise à jour des layouts
    fig.update_layout(
        plot_bgcolor=colors["background_color"],
        paper_bgcolor=colors["background_color"],
        font=dict(color=colors["text_color"], family="Inter"),
        xaxis=dict(
            title="Date",
            autorange=True,
            color=colors["text_color"],
            gridcolor=colors["dark_grey"],
            tickmode="auto",
            nticks=10,
            tickangle=45,
            # showline=False,
            zeroline=False,
            mirror=False,
            anchor="y",
        ),
        yaxis=dict(
            title="Prix ($)",
            color=colors["text_color"],
            gridcolor=colors["dark_grey"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_yaxes(showline=False)
    fig.update_yaxes(zeroline=False)

    return fig

df_backtest = pd.read_csv(os.path.join(DATA_DIR,"BTC_backtest_0.5.csv"))
fig_backtest = create_comparison_plot(df_backtest)

###### DATA functions ######

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_features(df):
    df = df.copy()
    
    # SMA
    df["sma_7d"] = df["close"].rolling(window=7*24).mean()
    df["sma_30d"] = df["close"].rolling(window=30*24).mean()
    df["sma_50d"] = df["close"].rolling(window=50*24).mean()
    df["sma_100d"] = df["close"].rolling(window=100*24).mean()
    
    # Volatibility
    df["return"] = df["close"].pct_change()

    # Volatility on 20 hours
    df["volatility_20"] = df["return"].rolling(window=20).std()
    df["volatility_50"] = df["return"].rolling(window=50).std()
    df["volatility_100"] = df["return"].rolling(window=100).std()
    df["volatility_14d"] = df["return"].rolling(window=14*24).std()
    
    # RSI 14 and 14 days
    df["rsi_14"] = compute_rsi(df["close"], window=14)
    df["rsi_14d"] = compute_rsi(df["close"], window=14*24)
    
    # MACD
    df = compute_MACD(df)
    
    # Relative volume 20
    df["volume_sma20"] = df["volume"].rolling(window=20).mean()
    df["volume_sma20d"] = df["volume"].rolling(window=20*24).mean()
    df["volume_rel20"] = df["volume"] / df["volume_sma20"]
    df["volume_rel20d"] = df["volume"] / df["volume_sma20d"]
    
    return df

def compute_MACD(df):
    df = df.copy()
    
    # EMA 12 et EMA 26
    df["ema_12d"] = df["close"].ewm(span=12*24, adjust=False).mean()
    df["ema_26d"] = df["close"].ewm(span=26*24, adjust=False).mean()
    
    # MACD line
    df["MACD"] = df["ema_12d"] - df["ema_26d"]

    # Signal line (EMA 9 du MACD)
    df["Signal"] = df["MACD"].ewm(span=9*24, adjust=False).mean()

    # Histogramme
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    return df

def clean_dataset(df):
    df = df.copy()
    
    # Colonnes inutiles
    drop_cols = [
        "close_time", "ignore", 
        "ema_12d", "ema_26d", 
        "volume_sma20", "volume_sma20d"
    ]
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Suppression des lignes avec NaN restants
    df = df.fillna(method="ffill")
    
    # # Dictionnaire : colonne long terme -> colonne court terme de fallback
    # fallback_map = {
    #     "sma_30d": "sma_7d",
    #     "sma_50d": "sma_7d",
    #     "sma_100d": "sma_7d",
    #     "volatility_50": "volatility_20",
    #     "volatility_100": "volatility_20",
    #     "volatility_14d": "volatility_20",
    #     "rsi_14d": "rsi_14",
    #     "volume_rel20d": "volume_rel20",
    # }

    # for long_col, short_col in fallback_map.items():
    #     if long_col in df.columns and short_col in df.columns:
    #         # Remplacer les NaN de la colonne long par les valeurs de la colonne short
    #         df[long_col] = df[long_col].fillna(df[short_col])

    # # Enfin, on propage dans le temps les éventuels NaN restants
    # df = df.ffill()
    df = df.fillna(0)
    
    return df

def reassign_close_to_open(df: pd.DataFrame) -> pd.DataFrame:
    """Reassign the 'open' price of each hour to be the 'close' price of the previous hour.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data and a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with updated 'open' prices.
    """
    df = df.copy()
    df["shifted_close"] = df["close"].shift(1)
    df["open"] = df["shifted_close"]
    df.drop(columns=["shifted_close"], inplace=True)
    # df.dropna(inplace=True)
    return df

def define_future_evolution(df: pd.DataFrame) -> pd.DataFrame:
    """Define a boolean target indicating if the price will go up in the next hour.

    Args:
        df (pd.DataFrame): DataFrame with OHLC data.

    Returns:
        pd.DataFrame: DataFrame with an additional 'will_up' column.
    """
    df = df.copy()
    df["will_up"] = (df["close"].shift(-1) > df["close"]).astype(bool)
    return df



