#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Performance Dashboard

This script provides a web dashboard to visualize model training results and performance metrics.
Uses Dash and Plotly for interactive visualization.
"""

import os
import sys
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'trained')

def load_metrics_files():
    """Load all metrics files"""
    metrics_files = glob.glob(os.path.join(MODELS_DIR, 'metrics_*.json'))
    metrics_data = []
    
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Add filename and timestamp
                file_name = os.path.basename(file_path)
                timestamp = datetime.strptime(file_name.split('_')[1].split('.')[0], '%Y%m%d_%H%M%S')
                data['file'] = file_name
                data['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                metrics_data.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return metrics_data

def load_window_metrics():
    """Load window metrics from CSV files"""
    window_files = glob.glob(os.path.join(MODELS_DIR, 'window_metrics_*.csv'))
    window_data = []
    
    for file_path in window_files:
        try:
            df = pd.read_csv(file_path)
            # Add filename and timestamp
            file_name = os.path.basename(file_path)
            timestamp = datetime.strptime(file_name.split('_')[2].split('.')[0], '%Y%m%d_%H%M%S')
            df['file'] = file_name
            df['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            window_data.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if window_data:
        return pd.concat(window_data, ignore_index=True)
    else:
        return pd.DataFrame()

def create_dashboard():
    """Create Dash dashboard"""
    # Initialize Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    
    # Define layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("SOL Trading Model Performance Dashboard", className="text-center mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Training Sessions"),
                    dbc.CardBody([
                        html.Div(id='session-list-container')
                    ])
                ])
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Overall Performance Metrics"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("Sharpe Ratio", className="card-title text-center"),
                                        html.H2(id="sharpe-value", className="text-center")
                                    ])
                                ])
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("Total Return", className="card-title text-center"),
                                        html.H2(id="return-value", className="text-center")
                                    ])
                                ])
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("Win Rate", className="card-title text-center"),
                                        html.H2(id="winrate-value", className="text-center")
                                    ])
                                ])
                            ], width=4),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("Max Drawdown", className="card-title text-center"),
                                        html.H2(id="drawdown-value", className="text-center")
                                    ])
                                ])
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("Sortino Ratio", className="card-title text-center"),
                                        html.H2(id="sortino-value", className="text-center")
                                    ])
                                ])
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("Profit Factor", className="card-title text-center"),
                                        html.H2(id="profit-factor-value", className="text-center")
                                    ])
                                ])
                            ], width=4),
                        ]),
                    ])
                ])
            ], width=8)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Training Performance"),
                    dbc.CardBody([
                        dcc.Graph(id='performance-chart', style={"height": "400px"})
                    ])
                ])
            ], width=12),
        ], className="mt-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Metrics by Window"),
                    dbc.CardBody([
                        dcc.Graph(id='window-metrics-chart', style={"height": "400px"})
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Metrics Correlation"),
                    dbc.CardBody([
                        dcc.Graph(id='metrics-correlation', style={"height": "400px"})
                    ])
                ])
            ], width=6),
        ], className="mt-4"),
        
        # Hidden div for storing the selected session data
        html.Div(id='selected-session-data', style={'display': 'none'}),
        
        dcc.Interval(
            id='interval-component',
            interval=30*1000,  # Update every 30 seconds
            n_intervals=0
        ),
    ], fluid=True)
    
    # Callbacks
    @app.callback(
        Output('session-list-container', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_session_list(n):
        """Update the list of training sessions"""
        metrics_data = load_metrics_files()
        
        if not metrics_data:
            return html.P("No training sessions found")
        
        # Sort by timestamp
        metrics_data.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Create list of sessions
        session_list = []
        for idx, session in enumerate(metrics_data):
            timestamp = session['timestamp']
            sharpe = session.get('sharpe_ratio', 'N/A')
            sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else sharpe
            
            session_list.append(
                dbc.ListGroupItem(
                    [
                        html.Div([
                            html.H5(f"Session {idx+1}: {timestamp}"),
                            html.P(f"Sharpe: {sharpe_str}")
                        ]),
                        html.Button("Select", id=f"select-session-{idx}", 
                                   className="btn btn-primary btn-sm", 
                                   **{'data-session-id': idx})
                    ],
                    className="d-flex justify-content-between align-items-center"
                )
            )
        
        return dbc.ListGroup(session_list)
    
    @app.callback(
        Output('selected-session-data', 'children'),
        Input('session-list-container', 'children'),
        prevent_initial_call=True
    )
    def handle_session_selection(children):
        """Handle session selection"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        
        # Get button ID
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if not button_id.startswith('select-session-'):
            return dash.no_update
        
        # Get session ID
        session_id = int(button_id.split('-')[-1])
        
        # Get metrics data
        metrics_data = load_metrics_files()
        metrics_data.sort(key=lambda x: x['timestamp'], reverse=True)
        
        if session_id < len(metrics_data):
            return json.dumps(metrics_data[session_id])
        
        return dash.no_update
    
    @app.callback(
        [
            Output('sharpe-value', 'children'),
            Output('return-value', 'children'),
            Output('winrate-value', 'children'),
            Output('drawdown-value', 'children'),
            Output('sortino-value', 'children'),
            Output('profit-factor-value', 'children'),
            Output('performance-chart', 'figure'),
            Output('window-metrics-chart', 'figure'),
            Output('metrics-correlation', 'figure')
        ],
        Input('selected-session-data', 'children'),
        prevent_initial_call=True
    )
    def update_dashboard(selected_session_data):
        """Update dashboard with selected session data"""
        if not selected_session_data:
            return dash.no_update
        
        # Parse session data
        session_data = json.loads(selected_session_data)
        
        # Update metrics cards
        sharpe = session_data.get('sharpe_ratio', 'N/A')
        sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else sharpe
        
        total_return = session_data.get('total_return', 'N/A')
        return_str = f"{total_return*100:.2f}%" if isinstance(total_return, (int, float)) else total_return
        
        win_rate = session_data.get('win_rate', 'N/A')
        win_rate_str = f"{win_rate*100:.2f}%" if isinstance(win_rate, (int, float)) else win_rate
        
        max_drawdown = session_data.get('max_drawdown', 'N/A')
        drawdown_str = f"{max_drawdown*100:.2f}%" if isinstance(max_drawdown, (int, float)) else max_drawdown
        
        sortino = session_data.get('sortino_ratio', 'N/A')
        sortino_str = f"{sortino:.2f}" if isinstance(sortino, (int, float)) else sortino
        
        profit_factor = session_data.get('profit_factor', 'N/A')
        profit_factor_str = f"{profit_factor:.2f}" if isinstance(profit_factor, (int, float)) else profit_factor
        
        # Create performance chart
        # This would normally show portfolio value over time
        # For now, we'll just create a placeholder chart
        perf_fig = go.Figure()
        perf_fig.add_trace(go.Scatter(x=[0, 1, 2], y=[1, 1.1, 1.2], mode='lines', name='Portfolio Value'))
        perf_fig.update_layout(
            title="Portfolio Performance",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Load window metrics
        window_metrics = load_window_metrics()
        
        # Create window metrics chart
        if not window_metrics.empty:
            # Filter by timestamp if needed
            timestamp = session_data.get('timestamp', None)
            if timestamp:
                window_metrics = window_metrics[window_metrics['timestamp'] == timestamp]
            
            window_fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            if 'sharpe_ratio' in window_metrics.columns:
                window_fig.add_trace(
                    go.Scatter(x=window_metrics['window'], y=window_metrics['sharpe_ratio'], 
                              mode='lines+markers', name='Sharpe Ratio'),
                    secondary_y=False
                )
            
            if 'win_rate' in window_metrics.columns:
                window_fig.add_trace(
                    go.Scatter(x=window_metrics['window'], y=window_metrics['win_rate'], 
                              mode='lines+markers', name='Win Rate'),
                    secondary_y=True
                )
            
            window_fig.update_layout(
                title="Metrics by Training Window",
                xaxis_title="Window",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            window_fig.update_yaxes(title_text="Sharpe Ratio", secondary_y=False)
            window_fig.update_yaxes(title_text="Win Rate", secondary_y=True)
        else:
            window_fig = go.Figure()
            window_fig.update_layout(
                title="No window metrics data available",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=40, b=40)
            )
        
        # Create correlation chart
        if not window_metrics.empty:
            # Calculate correlation
            metric_cols = [col for col in window_metrics.columns 
                         if col not in ['window', 'file', 'timestamp']]
            
            if metric_cols:
                corr = window_metrics[metric_cols].corr()
                
                corr_fig = px.imshow(
                    corr,
                    x=corr.columns,
                    y=corr.columns,
                    color_continuous_scale="RdBu_r",
                    text_auto=True
                )
                corr_fig.update_layout(
                    title="Metrics Correlation",
                    template="plotly_dark",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
            else:
                corr_fig = go.Figure()
                corr_fig.update_layout(
                    title="No correlation data available",
                    template="plotly_dark",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
        else:
            corr_fig = go.Figure()
            corr_fig.update_layout(
                title="No correlation data available",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=40, b=40)
            )
        
        return (
            sharpe_str,
            return_str,
            win_rate_str,
            drawdown_str,
            sortino_str,
            profit_factor_str,
            perf_fig,
            window_fig,
            corr_fig
        )
    
    return app

def main():
    """Main function"""
    app = create_dashboard()
    app.run_server(debug=True, port=8050)

if __name__ == "__main__":
    main() 