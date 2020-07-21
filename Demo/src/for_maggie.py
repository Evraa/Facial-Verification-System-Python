from auxilary import *

#to read data frame
main_dict = read_csv()

#to know how many images u have
print (len(main_dict))

#to find the main col labels
print (main_dict.columns)

#to extract features only
features = main_dict.columns[1:8] 
print (features)

#to iterate over them
for feature in features:
    print (feature)

#to find image[i]'s feature
for index, row in main_dict.iterrows():
    print (index) #like the id of the row
    # print(row) #the entire row
    print (row[features[0]])
    print (row[features[1]])
    print (row[features[2]])
    print (row[features[3]])
    print (row[features[4]])
    print (row[features[5]])
    print (row[features[6]])
    

