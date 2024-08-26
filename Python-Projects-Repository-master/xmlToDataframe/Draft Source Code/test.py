import lxml.etree as etree 

inputFile = input("Please enter your desired .xml file: ")
if len(inputFile) < 1 : inputFile = 'PtH_02.xml'
print(type(inputFile))

x = etree.parse(inputFile) 
print(etree.tostring(x, pretty_print = True))

