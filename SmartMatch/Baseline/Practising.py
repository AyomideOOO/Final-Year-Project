# This is my first draft of my baseline code
# The code will be object orientated later but in this file
# I will have the whole process to show the steps I took to implement 
# the problem 

import pandas as pd

# df -> data frame

df = pd.read_csv("data_set.csv")



# Mini practise Exercise
# Load the dataset.
# Print only the names of people whose Occupation contains "Likes".
# Count how many people have "Engineer" in their Job.
# Add a new column "Category":
# "Engineer" in Job → "STEM"
# otherwise → "Other".
# Print the updated dataframe.

#1)
# returns true or false if that rows occupation contains 'Likes'
likes_occ = df['Occupation'].str.contains("Likes" ,case = False)



#prints the names of values that are true.
print(df.loc[likes_occ,'Name'])


#2)
# Gets the sum of engineering jobs in the dataset. 
engineer_count = df['Job'].str.contains("Engineer", case = False).sum()
print(f'The sum of jobs with Engineering is: {engineer_count}')


#3)
# creating a new column for my dataFrame
# made the catgeories the same because category is based of job but applied the function lambda
# which: STEM if 'Engineer' is in x else Other.
df['Category'] = df['Job'].apply(lambda x: 'STEM' if 'Engineer' in x else 'Other')

print(df)

# Start implementation of BASELINE 



