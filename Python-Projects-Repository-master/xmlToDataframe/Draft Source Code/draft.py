import pandas as pd

dict =  {'scene': ["foul", "murder", "drunken", "intrigue", ""],
'facade': ["fair", "beaten", "fat", "elf", ""], 'maraki': ["", "", "","","gamotospitisou"]}

idx = ['hamlet', 'lear', 'falstaff','puck', 'christos']
dp = pd.DataFrame(dict,index=idx)

print(dp)