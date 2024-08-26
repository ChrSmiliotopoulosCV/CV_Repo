# ################ Data on the Web ################
# With the HTTP Request/Respond well understood and well
# supported, there was a natural move toward exchanging 
# data between programs using these protocols. 

# We needed to come up with an agreed way to represent data
# going between applications and across networks. 

# There are two commonly used formats: XML and JSON.

# ################ Agreed on a "Wire Format" ################
# Serialize - De-Serialize

# ############# XML - eXtensible Markup Language #############
# ####################### XML - Basics #######################
# Start Tag, End Tag, Text Content, Attribute, Self Closing Tag

# <person> --- Start Tag
#  <name>Christos</name> --- Attribute
#  <phone type="intl">
#   +306947800766 --- Text Content
#  </phone>
#  <email hide="yes"/> --- Self Closing Tag
# </person> --- End Tag

# All the aboves are called XML elements or "nodes" (XML can be though as
# a tree, as a text and attributes or as paths.)

import xml.etree.ElementTree as ET

data = '''
<person>
 <name>Christos</name> 
 <phone type="intl">
  +306947800766 
 </phone>
 <email hide="yes"/> 
</person>
'''
tree = ET.fromstring(data)
print('Name:', tree.find('name').text)
print('Attr:', tree.find('email').get('hide'))

input = '''
<stuff>
  <users>
    <user x="2">
      <id>001</id>
      <name>Chuck</name>
    </user>
    <user x="7">
      <id>009</id>
      <name>Brent</name>
    </user>
  </users>
</stuff>'''

stuff = ET.fromstring(input)
lst = stuff.findall('users/user')
print('User count:', len(lst))

for item in lst:
    print('Name', item.find('name').text)
    print('Id', item.find('id').text)
    print('Attribute', item.get('x'))

# ################ XML Schema ################
# Describing a "contract" as to what is acceptable XML.

# Description of the legal format of an XML document. 

# Expressed in terms of constraints on the structure and 
# content of documents. 

# Often used to specify a "contract" between systems - "My system
# will only accept XML that conforms to this particular Schema".

# If a particular piece of XML meets the specification of the Schema - 
# it is said to "validate".

# There are many XML schemas. DTD, SGML, W3C. We are going to deal with the 
# W3C XML schema. It is often called XSD because the file names end in .xsd.






