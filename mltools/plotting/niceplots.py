import matplotlib.pyplot as plt
import numpy as np

color_1 = np.array([255, 105, 180, 0.6 * 255]) / 255
color_2 = np.array([0, 191, 255, 0.6 * 255]) / 255

def get_linear_colors(n):
    return np.linspace(color_1, color_2, n)

def apply_global_settings():
    plt.rcParams['figure.figsize'] = [8,6]
    plt.rcParams['figure.dpi'] = 200
    #font 25
    plt.rcParams['font.size'] = 25

def apply_plot_style(fig, ax,lw=3,labelsize=25):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    fig.tight_layout()

    #get all lines
    lines = ax.get_lines()
    for line in lines:
        line.set_linewidth(lw)

def apply_heatmap_style(fig, ax,labelsize=25):
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Remove ticks
    ax.tick_params(axis='both', which='both', length=0)
    # Font sizes
    ax.xaxis.label.set_fontsize(labelsize)
    ax.yaxis.label.set_fontsize(labelsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    
    # Layout
    fig.tight_layout()