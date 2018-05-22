print(__doc__)

import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

lw = 2

trace1 = go.Scatter(x=fpr[2], y=tpr[2],
                    mode='lines',
                    line=dict(color='darkorange', width=lw),
                    name='ROC curve (area = %0.2f)' % roc_auc[2]
                    )

trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(color='navy', width=lw, dash='dash'),
                    showlegend=False)

layout = go.Layout(title='Receiver operating characteristic example',
                   xaxis=dict(title='False Positive Rate'),
                   yaxis=dict(title='True Positive Rate'))

fig = go.Figure(data=[trace1, trace2], layout=layout)
