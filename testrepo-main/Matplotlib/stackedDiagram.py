import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# create data
df = pd.DataFrame(
    [
        ["ProcessId", 0.938707, 0.131629],
        ["CompSTA1", 0.671768, 0.97732],
        ["SysTimeMonth_6", 0.671768, 0.97732],
        ["SysTimeWeek_25", 0.671768, 0.97732],
        ["SysTimeYear_2022", 0.644611, 0.52697],
        ["DstPortName_netbios-dgm", 0.614198, 0.425863],
        ["Init_True", 0.581108, 0.361841],
        ["DstPortName_microsoft-ds", 0.555811, 0.108113],
        ["SysTimeday", 0.428025, 0.660785],
        ["EventID5", 0.424617, 0.066464],
        ["SysTimeMonth_12", 0.406318, 0.396795],
        ["DstPortName_netbios-ns", 0.34378, 0.282503],
        ["EventID3", 0.324176, 0.685845],
        ["EventID13", 0.314004, 0.216361],
        ["CompSTA3", 0.302525, 0.387711],
        ["CompSTA2", 0.299937, 0.506158],
        ["SysTimeWeek_48", 0.279761, 0.517776],
        ["EventID22", 0.251431, 0.177107],
        ["EventID11", 0.231781, 0.152359],
        ["SysTimeWeek_46", 0.224268, 0.088499],
        ["EventID17", 0.191692, 0.01457],
        ["SysTimeYear_2021", 0.182975, 0.54303],
        ["EventID18", 0.181182, 0.022548],
        ["SysTimeMonth_1", 0.175161, 0.24006],
        ["SysTimeYear_2023", 0.175161, 0.24006],
        ["EventID1", 0.155295, 0.313569],
        ["SysTimeMonth_3", 0.152727, 0.170713],
        ["SysTimeWeek_10", 0.152727, 0.170713],
        ["SysTimeWeek_2", 0.151823, 0.194818],
        ["SysTimeDoW_5", 0.148806, 0.384821],
        ["SysTimeminute", 0.094598, 0.087914],
        ["SysTimeWeek_1", 0.089097, 0.139639],
        ["SysTimeWeek_49", 0.078424, 0.197668],
        ["SysTimeWeek_47", 0.077554, 0.122149],
        ["EventID15", 0.057299, 0.028739],
        ["SysTimehour", 0.049749, 0.179999],
        ["EventID12", 0.048199, 0.042263],
        ["EventID8", 0.046988, 0.02782],
        ["SrcIpv6_True", 0.044832, 0.344724],
        ["SysTimeDoW6", 0.043931, 0.117577],
        ["SysTimeWeek_50", 0.040381, 0.170286],
        ["EventID16", 0.040285, 0.001305],
        ["EventID4", 0.033271, 0.009152],
        ["SysTimeDoW_0", 0.031521, 0.110259],
        ["EventID2", 0.02807, 0.020793],
        ["SysTimeMonth_11", 0.020581, 0.455022]
        # ,
        # ["EventID255", 0.014853, 0.004293]
        # ,
        # ["EventID6", 0.012909, 0.009616],
        # ["DstPortName_ntp", 0, 0],
    ],
    columns=["Feature Importance Method", "Coef", "PCA"],
)
# view data
print(df)

# plot data in stack manner of bar type
df.plot(x="Feature Importance Method", kind="bar", stacked=True)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Feature Importance Score', fontsize=15)
plt.subplots_adjust(bottom=0.25)
plt.margins(0.02)
# plt.ylim(bottom=-1.5, top=1.2)
plt.xticks(rotation=35, fontsize=8, ha="right")
# plt.xticks(rotation='vertical')
plt.show()
