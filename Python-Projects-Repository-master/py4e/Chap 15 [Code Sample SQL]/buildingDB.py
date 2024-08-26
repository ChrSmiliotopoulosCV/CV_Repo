# ########## Building a Data Model ##########
# Drawing a picture of the data objects for our application and then 
# figuring our how to represent the objects and their relationships.

# Basic Rule: Don't put the same string data in twice - use a relationship
# instead. When there is one thing in the "real world" there should be one
# copy of that thing in the database.

# For each piece of info...
# Is the column an object or an attribute of another object?
# Once we define objects, we need to define the relationships between the 
# objects.

# ########## Representing Relationships in a Database ##########
# A trick is to look for any dublicate more than one times stings or integer.
# In that case we put each multiply repeatitive string or integer to each own 
# table, which will then be related to each other. 

# ########## Database Normalization ##########
# There is tons of database theory - way too much to understand without excessive
# predictive calculus.
# Do not replicate data - reference data - point at data.
# Use integers for keys and for references.
# Add a special "key" column to each table which we will make references to. By 
# convention, many programmers call this column "id".

# ########## Three kinds of keys ##########
# Primary key: generally an integer auto-increment field.
# Logical key: What the outside world uses for lookup.
# Foreign key: generally an integer key pointing to a row in another table.

# ########## Kay Rules ##########
# Best Practices:
# Never use your logical key as the primary key.
# Logical keys can and do change, albeit slowly.
# Relationships that are based on matching string are less efficient than integers.

# ########## Basic Convention during Database Construction ##########
# Table title: Always starts with uppercase letter.
# Primary key: Always is an integer incremental id.
# Foreign key: Is always represented as artist_id.

# ########## The JOIN Operation ##########
# The JOIN operation links across several tables as part of a select operation.
# You must tell the JOIN how to use the keys that make the connection between the tables
# using an ON clause.

