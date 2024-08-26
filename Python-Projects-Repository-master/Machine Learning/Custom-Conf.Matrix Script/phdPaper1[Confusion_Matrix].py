import seaborn as sns
import matplotlib.pyplot as plt   
import numpy as np
from sqlalchemy import true  

# Input data for the Confusion Matrices of Table 7
# cm = np.array([[70641, 7224],[1204, 2006]])
# cm = np.array([[250666,   23481],[3815,  15556]])
# cm = np.array([[347384,   47843],[7072,  13728]])
# cm = np.array([[736120,   100993],[12181,  20882]])

# Input data for the Confusion Matrices of Table 7
# cm = np.array([[71949,   1083],[1139,  6100]])
# cm = np.array([[260205,   5576],[3669,  24068]])
# cm = np.array([[367145,   9568],[6656,  32658]])
cm = np.array([[770490,   19142],[11746,  68739]])

sns.set(font_scale = 1.75)
x_axis_labels = ["Positive", "Negative"] # labels for x-axis
y_axis_labels = ["Positive", "Negative"] # labels for y-axis
p = sns.heatmap(cm, annot=True,fmt="d",cmap='Blues', cbar=False, annot_kws={"size":30})
p.xaxis.tick_top() # x axis on top
p.xaxis.set_label_position('top')
p.tick_params(length=0)
p.set( xlabel = "Predicted label", ylabel = "True label", xticklabels = x_axis_labels, yticklabels = y_axis_labels)

plt.show()

# Table 7 Subsets Results
# Normal subset
# ---TP-------FP-----FN-----TN---    
# cm = np.array([[70641,   1204],[7224,  2006]])

# NormalVsMalicious01 subset
# cm = np.array([[250666,   23481],[3815,  15556]])

# NormalVsMalicious02 subset
# cm = np.array([[347384,   47843],[7072,  13728]])

# FullSet subset
# cm = np.array([[736120,   100993],[12181,  20882]])

# Table 8 Subsets Results
# Normal subset
# ---TP-------FP-----FN-----TN---    
# cm = np.array([[71949,   1083],[1139,  6100]])

# NormalVsMalicious01 subset
# cm = np.array([[260205,   5576],[3669,  24068]])

# NormalVsMalicious02 subset
# cm = np.array([[367145,   9568],[6656,  32658]])

# FullSet subset
# cm = np.array([[770490,   19142],[11746,  68739]])