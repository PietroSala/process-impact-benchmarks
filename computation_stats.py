import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


def create_log_heatmap(df, x_col, y_col, value_col, title, use_std=False, colorscale='Viridis', colorbar_x=0, colorbar_y=1.1, log=True):
    # Group by x and y columns, then calculate the average or standard deviation
    if use_std:
        agg_data = df.groupby([x_col, y_col])[value_col].std().reset_index()
        metric_label = 'Std Dev'
    else:
        agg_data = df.groupby([x_col, y_col])[value_col].mean().reset_index()
        metric_label = 'Avg'
    
    # Create a pivot table for aggregated values
    pivot_data = agg_data.pivot(index=y_col, columns=x_col, values=value_col)


    # Apply log10 to the values for color scaling, handling zeros and negative values
    log_values = np.log10(pivot_data.values, where=(pivot_data.values > 0)) if log else pivot_data.values
    log_values[np.isinf(log_values)] = np.nan    # Replace inf (log of 0) with nan

    # Create the heatmap with log values for color scaling, but show aggregated values
    heatmap = go.Heatmap(
        z=log_values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=colorscale,
        colorbar=dict(
            title=f'{"Log10" if log else ""} {metric_label} Time (s)',
            tickprefix="10^" if log else "",
            len=0.4,
            yanchor="bottom",
            y= colorbar_y,
            x= colorbar_x,
            xanchor="right",
            orientation="h"
        ),
        hovertemplate=(f'{y_col}: %{{y}}<br>{x_col}: %{{x}}<br>' +
                       f'{metric_label} Time: %{{text}} s<br>' +
                       f'{"Log10" if log else ""} {metric_label} Time: %{{z:.2f}}<extra></extra>'),
        text=[[f'{val:.2e}' for val in row] for row in pivot_data.values],
        texttemplate="%{text}",
        zmin=np.nanmin(log_values), 
        zmax=np.nanmax(log_values), 
        zauto= False,
        colorbar_tickformat=".1f" if log else ".1e",
    )

    return heatmap

def create_log_heatmaps(df, x_col, y_col, value_col, title, height=600, width=1000, colorbar_x_left=0, colorbar_x_right=0, colorbar_y_left=0, colorbar_y_right=0, log=True):

    # Create subplots
    fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=(f"Average {title} (Log10 Scale)", 
                                    f"Standard Deviation of {title} (Log10 Scale)") if log else (f"Average {title}", f"Standard Deviation of {title}"),
                    shared_yaxes=True,
                    horizontal_spacing=0.1)

    # Create and add heatmaps
    avg_heatmap = create_log_heatmap(df, x_col, y_col, value_col, 'Average', use_std=False, colorscale='Viridis', colorbar_x=colorbar_x_left, colorbar_y=colorbar_y_left, log=log)
    std_heatmap = create_log_heatmap(df, x_col, y_col, value_col, 'Std Dev', use_std=True, colorscale='Plasma', colorbar_x=colorbar_x_right, colorbar_y=colorbar_y_right, log=log)

    fig.add_trace(avg_heatmap, row=1, col=1)
    fig.add_trace(std_heatmap, row=1, col=2)

    # Update layout
    fig.update_layout(
        height=height,  # Increased height to accommodate colorbars below
        width=width,
        title_text=f"{title}: Average and Standard Deviation",
        xaxis=dict(title=x_col, dtick=1),
        xaxis2=dict(title=x_col, dtick=1),
        yaxis=dict(title=y_col, dtick=1)
    )

    # Update traces to ensure text is visible
    fig.update_traces(textfont_color='black', textfont_size=9)

    # Show the figure
    fig.show()

def missing_values(df, nested=None, independent=None, process_number=None, dimensions=None, 
                   modes=["random", "bagging_divide", "bagging_remove","bagging_remove_divide","bagging_remove_reverse", "bagging_remove_reverse_divide"]):
    for i in tqdm(range(1, nested + 1 if nested else max(df['nested'])+1)):
        for j in range(1, independent + 1 if independent else max(df['independent'])+1):
            for l in range(1, process_number + 1 if process_number else max(df['process_number'])+1):
                for d in range(1, dimensions + 1 if dimensions else max(df['dimensions'])+1):  
                    for m in  modes:
                        if not ((df['nested'] == i) & (df['independent'] == j) & (df['process_number'] == l) & (df['dimensions'] == d) & (df['mode'] == m)).any():
                            print(f"Missing values for nested={i}, independent={j}, process_number={l}, dimensions={d}, mode={m}")