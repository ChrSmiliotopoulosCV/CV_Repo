# Importing Libraries
import numpy as np
import pandas as pd

# Python statement to handle the pandas chained_assignment warning exception.
pd.options.mode.chained_assignment = None  # default='warn'

# Importing the Dataset.
# Read data from CSV file into pandas dataframe and import the column names to it from AWID2 paper.
awid2df = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Datasets\DATASET\DATASET\AWID-CLS-F-Tst\12.csv"),
    names=["frame.interface_id", "frame.dlt", "frame.offset_shift", "frame.time_epoch", "frame.time_delta",
           "frame.time_delta_displayed", "frame.time_relative", "frame.len", "frame.cap_len", "frame.marked",
           "frame.ignored", "radiotap.version", "radiotap.pad", "radiotap.length", "radiotap.present.tsft",
           "radiotap.present.flags", "radiotap.present.rate", "radiotap.present.channel", "radiotap.present.fhss",
           "radiotap.present.dbm_antsignal", "radiotap.present.dbm_antnoise", "radiotap.present.lock_quality",
           "radiotap.present.tx_attenuation", "radiotap.present.db_tx_attenuation", "radiotap.present.dbm_tx_power",
           "radiotap.present.antenna", "radiotap.present.db_antsignal", "radiotap.present.db_antnoise",
           "radiotap.present.rxflags", "radiotap.present.xchannel", "radiotap.present.mcs", "radiotap.present.ampdu",
           "radiotap.present.vht", "radiotap.present.reserved", "radiotap.present.rtap_ns",
           "radiotap.present.vendor_ns",
           "radiotap.present.ext", "radiotap.mactime", "radiotap.flags.cfp", "radiotap.flags.preamble",
           "radiotap.flags.wep", "radiotap.flags.frag", "radiotap.flags.fcs", "radiotap.flags.datapad",
           "radiotap.flags.badfcs", "radiotap.flags.shortgi", "radiotap.datarate", "radiotap.channel.freq",
           "radiotap.channel.type.turbo", "radiotap.channel.type.cck", "radiotap.channel.type.ofdm",
           "radiotap.channel.type.2ghz", "radiotap.channel.type.5ghz", "radiotap.channel.type.passive",
           "radiotap.channel.type.dynamic", "radiotap.channel.type.gfsk", "radiotap.channel.type.gsm",
           "radiotap.channel.type.sturbo", "radiotap.channel.type.half", "radiotap.channel.type.quarter",
           "radiotap.dbm_antsignal", "radiotap.antenna", "radiotap.rxflags.badplcp", "wlan.fc.type_subtype",
           "wlan.fc.version", "wlan.fc.type", "wlan.fc.subtype", "wlan.fc.ds", "wlan.fc.frag", "wlan.fc.retry",
           "wlan.fc.pwrmgt", "wlan.fc.moredata", "wlan.fc.protected", "wlan.fc.order", "wlan.duration", "wlan.ra",
           "wlan.da", "wlan.ta", "wlan.sa", "wlan.bssid", "wlan.frag", "wlan.seq", "wlan.bar.type",
           "wlan.ba.control.ackpolicy", "wlan.ba.control.multitid", "wlan.ba.control.cbitmap",
           "wlan.bar.compressed.tidinfo", "wlan.ba.bm", "wlan.fcs_good", "wlan_mgt.fixed.capabilities.ess",
           "wlan_mgt.fixed.capabilities.ibss", "wlan_mgt.fixed.capabilities.cfpoll.ap",
           "wlan_mgt.fixed.capabilities.privacy", "wlan_mgt.fixed.capabilities.preamble",
           "wlan_mgt.fixed.capabilities.pbcc", "wlan_mgt.fixed.capabilities.agility",
           "wlan_mgt.fixed.capabilities.spec_man", "wlan_mgt.fixed.capabilities.short_slot_time",
           "wlan_mgt.fixed.capabilities.apsd", "wlan_mgt.fixed.capabilities.radio_measurement",
           "wlan_mgt.fixed.capabilities.dsss_ofdm", "wlan_mgt.fixed.capabilities.del_blk_ack",
           "wlan_mgt.fixed.capabilities.imm_blk_ack", "wlan_mgt.fixed.listen_ival", "wlan_mgt.fixed.current_ap",
           "wlan_mgt.fixed.status_code", "wlan_mgt.fixed.timestamp", "wlan_mgt.fixed.beacon", "wlan_mgt.fixed.aid",
           "wlan_mgt.fixed.reason_code", "wlan_mgt.fixed.auth.alg", "wlan_mgt.fixed.auth_seq",
           "wlan_mgt.fixed.category_code", "wlan_mgt.fixed.htact", "wlan_mgt.fixed.chanwidth",
           "wlan_mgt.fixed.fragment",
           "wlan_mgt.fixed.sequence", "wlan_mgt.tagged.all", "wlan_mgt.ssid", "wlan_mgt.ds.current_channel",
           "wlan_mgt.tim.dtim_count", "wlan_mgt.tim.dtim_period", "wlan_mgt.tim.bmapctl.multicast",
           "wlan_mgt.tim.bmapctl.offset", "wlan_mgt.country_info.environment", "wlan_mgt.rsn.version",
           "wlan_mgt.rsn.gcs.type", "wlan_mgt.rsn.pcs.count", "wlan_mgt.rsn.akms.count", "wlan_mgt.rsn.akms.type",
           "wlan_mgt.rsn.capabilities.preauth", "wlan_mgt.rsn.capabilities.no_pairwise",
           "wlan_mgt.rsn.capabilities.ptksa_replay_counter", "wlan_mgt.rsn.capabilities.gtksa_replay_counter",
           "wlan_mgt.rsn.capabilities.mfpr", "wlan_mgt.rsn.capabilities.mfpc", "wlan_mgt.rsn.capabilities.peerkey",
           "wlan_mgt.tcprep.trsmt_pow", "wlan_mgt.tcprep.link_mrg", "wlan.wep.iv", "wlan.wep.key", "wlan.wep.icv",
           "wlan.tkip.extiv", "wlan.ccmp.extiv", "wlan.qos.tid", "wlan.qos.priority", "wlan.qos.eosp", "wlan.qos.ack",
           "wlan.qos.amsdupresent", "wlan.qos.buf_state_indicated", "wlan.qos.bit4", "wlan.qos.txop_dur_req",
           "wlan.qos.buf_state_indicated", "data.len", "class"],
    error_bad_lines=False, encoding='ISO-8859-1', low_memory=False)

# See the rows and columns of the dataset.
awid2df.shape

# Depict how our dataset actually looks
awid2df.head()

# Print() statements to check whether the column names depict the right values.
# print(awid2df["frame.len"])
# print(awid2df["wlan.fc.pwrmgt"])

# New newAwid2df dataframe with only the 17 selected features included the 'class' column.
newAwid2df = awid2df[
    ['frame.len', 'radiotap.length', 'radiotap.present.tsft', 'radiotap.channel.freq', 'radiotap.channel.type.cck',
     'radiotap.channel.type.ofdm', 'radiotap.dbm_antsignal', 'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.ds',
     'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected', 'wlan.duration',
     'class']]

# Print the dataframe's total number_of_rows
index = newAwid2df.index
number_of_rows = len(index)
print("newAwid2df dataframe's total number_of_rows is:")
print(number_of_rows)

# newAwid2df isnull().any() null values identification and counting with sum()
print("isnull() check for newAwid2df dataframe:")
print(newAwid2df.isnull().any())
print("Number of null values in newAwid2df dataframe:")
print(newAwid2df.isnull().any().sum())

# newAwid2df.dropna() will remove the null, NaN and empty values from the dataframe.
newAwid2df = newAwid2df.dropna()
print("isnull() check for newAwid2df dataframe after dropna():")
print(newAwid2df.isnull().any())
print("The number of null values in newAwid2df dataframe after the dropna() has:")
print(newAwid2df.isnull().any().sum())

# Print the dataframe's new total number_of_rows after the deletion of the rows with null values.
index1 = newAwid2df.index
number_of_rows1 = len(index1)
print("newAwid2df dataframe's total number_of_rows after the deletion of the rows with null values:")
print(number_of_rows1)

# Two different print() functions to count the number of newAwid2df['radiotap.dbm_antsignal'] == "?" characters.
print("Number of ? characters in newAwid2df['radiotap.dbm_antsignal']:")
print(len(newAwid2df[newAwid2df['radiotap.dbm_antsignal'] == "?"]))
print((newAwid2df['radiotap.dbm_antsignal'] == "?").sum())
print("normal traffic:")
print((newAwid2df['class'] == "normal").sum())
print("injection traffic:")
print((newAwid2df['class'] == "injection").sum())
print("flooding traffic:")
print((newAwid2df['class'] == "flooding").sum())
print("impersonation traffic:")
print((newAwid2df['class'] == "impersonation").sum())

# Replace the "?" of newAwid2df['radiotap.dbm_antsignal'] with np.nan null values.
newAwid2df['radiotap.dbm_antsignal'].loc[newAwid2df['radiotap.dbm_antsignal'] == "?"] = np.nan
newAwid2df['wlan.duration'].loc[newAwid2df['wlan.duration'] == "?"] = np.nan
newAwid2df['class'].loc[newAwid2df['class'] == "injection"] = np.nan

# print() and sum() again the number of "?" characters in newAwid2df['radiotap.dbm_antsignal'] (it should be 0!!!)
print("Double check if any ? character has been left:")
print((newAwid2df['radiotap.dbm_antsignal'] == "?").sum())
print((newAwid2df['wlan.duration'] == "?").sum())
print("Double check if any injection class has been left:")
print(newAwid2df['class'].loc[newAwid2df['class'] == "injection"].sum())

# newAwid2df isnull().any() null values identification and counting with sum()
print("isnull() check for newAwid2df dataframe:")
print(newAwid2df.isnull().any())
print("Number of null values in newAwid2df dataframe:")
print(newAwid2df.isnull().any().sum())

# newAwid2df.dropna() will remove the null, NaN and empty values from the dataframe.
newAwid2df = newAwid2df.dropna()
print("isnull() check for newAwid2df dataframe after dropna():")
print(newAwid2df.isnull().any())
print("The number of null values in newAwid2df dataframe after the dropna() has:")
print(newAwid2df.isnull().any().sum())

# Print the dataframe's new total number_of_rows after the deletion of the rows with null values.
index2 = newAwid2df.index
number_of_rows2 = len(index2)
print("newAwid2df dataframe's total number_of_rows2 after the deletion of the rows with null values:")
print(number_of_rows2)

# newAwid2df['radiotap.dbm_antsignal'] pd.to_numeric() function to transform all the column values to numerical.
# abs() function to keep only the absolute values in the newAwid2df['radiotap.dbm_antsignal'] column.
newAwid2df['radiotap.dbm_antsignal'] = pd.to_numeric(newAwid2df['radiotap.dbm_antsignal'])
newAwid2df['radiotap.dbm_antsignal'] = newAwid2df['radiotap.dbm_antsignal'].abs()

# newAwid2df['wlan.fc.ds'] replacement of the prefixes with the relevant value.
newAwid2df.replace({'wlan.fc.ds': {'0x00': 0, '0x01': 1, '0x02': 2, '0x03': 3}}, inplace=True)

# print() function of the newAwid2df["class"].unique() unique values. This statement is clearly related to the
# dataset labelling of the traffic to normal, flooding, impersonation and injection.
print("Unique classes in newAwid2df['class'] column:")
print(newAwid2df["class"].unique())

# Make Normal class equal to 0, Flooding class equal to 1, and Impersonation class equal to 2
newAwid2df['class'] = newAwid2df['class'].map(str)
newAwid2df.replace({'class': {'normal': 0, 'flooding': 1, 'impersonation': 2}}, inplace=True)

# Transform to type float the newAwid2df['radiotap.dbm_antsignal'] and newAwid2df['wlan.duration'] columns.
newAwid2df['radiotap.dbm_antsignal'] = newAwid2df['radiotap.dbm_antsignal'].astype(float)
newAwid2df['wlan.duration'] = newAwid2df['wlan.duration'].astype(float)

# to_csv() function to write the newly manipulated dataframe to a .csv file.
newAwid2df.to_csv(
    "D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Datasets\DATASET\DATASET\AWID-CLS-F-Tst\Awid2dfTst12.csv",
    index=False)
