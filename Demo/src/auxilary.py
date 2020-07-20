import pandas as pd

my_dict = { 'name' : ["a", "b", "c", "d", "e","f", "g"],
                   'age' : [20,27, 35, 55, 18, 21, 35],
                   'designation': ["VP", "CEO", "CFO", "VP", "VP", "CEO", "MD"]}


# df = pd.DataFrame(my_dict)
# df.to_csv('csv_example.csv', index=False)
df = pd.read_csv('Demo/src/csv_example.csv')
df_csv = pd.read_csv('Demo/src/csv_example.csv', header=[1,2])
print(df_csv)



'''
TODO: convert csv to excel
'''