import numpy as np
from torch.utils.data import Dataset
import seaborn as sns
import matplotlib.pyplot as plt

########################################################################################################################
class TrainDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, indices):
        return self.x_data[indices], self.y_data[indices]


########################################################################################################################
def ruleHistoryPlotV2(trainer, num_rule, style='whitegrid', figsize=(10,6), threshold=1e-4):
    if isinstance(num_rule, int):
        num_rule = 'rule_' + str(num_rule)

    array = trainer.get_weight_history(num_rule).detach().cpu().numpy()

    # Create a Seaborn lineplot
    sns.set(style=style)  # Set the style if you prefer

    plt.figure(figsize=figsize)  # Set the figure size

    for neuron in range(array.shape[1]):
        non_zero_check = np.any(abs(array[:, neuron, :]) > threshold, axis=0)
        columns_with_non_zero = np.where(non_zero_check)[0]
        for col in columns_with_non_zero:
            sns.lineplot(x=range(array.shape[0]), y=array[:, neuron, col],
                         label=f'Neuron number {neuron + 1}, feature number {col + 1}')

    plt.xlabel("Num epoch")
    plt.ylabel("Value")
    plt.title(num_rule + ' history')

    plt.show()

########################################################################################################################
def AlmostRuleHistoryPlotV3(trainer, num_rule, threshold=1e-4, burn_in=0, background_color='white',
                            grid_color='darkgray', figsize=(1300, 900)):
    if isinstance(num_rule, int):
        num_rule = 'rule_' + str(num_rule)

    array = trainer.get_weight_history(num_rule, apply_act=False).detach().cpu().numpy()

    # Create an empty Plotly figure
    fig = go.Figure()

    for neuron in range(array.shape[1]):
        non_zero_check = np.any(abs(array[burn_in:, neuron, :]) > threshold, axis=0)
        columns_with_non_zero = np.where(non_zero_check)[0]
        for col in columns_with_non_zero:
            # Add a trace for each line plot
            trace = go.Scatter(x=list(range(array.shape[0])), y=array[:, neuron, col],
                               mode='lines',
                               name=f'Neuron {neuron + 1}, feature {col + 1}')
            fig.add_trace(trace)

    # Customize the layout
    fig.update_layout(
        xaxis_title="Num epoch",
        yaxis_title="Value",
        title=num_rule + ' history',
        plot_bgcolor=background_color,
        width=figsize[0],  # Set the width based on figsize
        height=figsize[1],
    )

    if grid_color is not None:
        # Modify the grid color
        fig.update_xaxes(showgrid=True, gridcolor=grid_color)
        fig.update_yaxes(showgrid=True, gridcolor=grid_color)
        fig.update_xaxes(zeroline=True, zerolinecolor=grid_color)
        fig.update_yaxes(zeroline=True, zerolinecolor=grid_color)

    # Show the interactive plot
    fig.show()