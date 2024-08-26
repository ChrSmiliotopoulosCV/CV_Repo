
import xml.etree.ElementTree as et


xtree = et.parse('xmlFile.xml')
xroot = xtree.getroot()

for node in xroot:
    s_name = node.attrib.get("name")
    # print(s_name)
    s_mail = node.find("email").text
    # print(s_mail)
    s_grade = node.find("grade").text
    # print(s_grade)
    s_age = node.find("age").text
    # print(s_age)

