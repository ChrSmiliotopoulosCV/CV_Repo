import xml.etree.ElementTree as et

# We can import this data by reading from a file:
xtree = et.parse('sysmon_WM_02.xml')
# print(type(xtree))
xroot = xtree.getroot()

# Or pass it in a variable with tostring() function:
xmlstr = et.tostring(xroot)
# print(xmlstr)

# And take it directly from a string:
root = et.fromstring(xmlstr)

for child in xroot:
    print(child.tag, child.attrib, child.text)
    if child.tag == '{http://schemas.microsoft.com/win/2004/08/events/event}RenderingInfo':
        print('OK')
        print(child.tag)
        xroot.remove(child)
        print('OK')
    print(child.tag)
    # for ch in child:
    #     # print(ch.tag)
    #     if ch.tag == '{http://schemas.microsoft.com/win/2004/08/events/event}Message':
    #         # for k, v in (ch.attrib).items():
    #         #     print(k)
    #         #     print(v)
    #         ch.text = ''
    #         print(ch.text)

# for EventID in root.iter('EventID'):
#      new_desc = str(EventID.text)+''
#      EventID.text = str(new_desc)
    #  description.set('updated', 'yes')
 
xtree.write('sysmon_WM_002.xml')