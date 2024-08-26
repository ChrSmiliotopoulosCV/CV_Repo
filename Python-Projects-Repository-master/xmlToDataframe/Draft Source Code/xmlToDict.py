#importing the modules
import xmltodict
import json
import pandas as pd
# #declaring the xml
# my_xml = """
#                 <System>
#                         <Provider Name="Microsoft-Windows-Sysmon" Guid="{5770385f-c22a-43e0-bf4c-06f5698ffbd9}"/>
#                         <EventID>3</EventID>
#                         <Version>5</Version>
#                         <Level>4</Level>
#                         <Task>3</Task>
#                         <Opcode>0</Opcode>
#                         <Keywords>0x8000000000000000</Keywords>
#                         <TimeCreated SystemTime="2022-06-25T16:58:58.6866018Z"/>
#                         <EventRecordID>4482</EventRecordID>
#                         <Correlation/>
#                         <Execution ProcessID="5520" ThreadID="6760"/>
#                         <Channel>Microsoft-Windows-Sysmon/Operational</Channel>
#                         <Computer>LAPTOP-ROPR18AK</Computer>
#                         <Security UserID="S-1-5-18"/>
#                 </System>
# """
# #coverting xml to Python dictionary
# dict_data = xmltodict.parse(my_xml)
# #coverting to json
# json_data = json.dumps(dict_data, indent=2)
# print(json_data)

df = pd.read_xml('PtH_02.xml')
print(df)
df.to_csv('output1.csv')