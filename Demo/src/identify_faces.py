from auxilary import *


def identify_faces_based_on_lengths():
    #read the data
    path_to_csv_lengths = "../csv_files/csv_lengths.csv"
    main_dict = read_csv(path_to_csv_lengths)
    if len(main_dict) == 0:
        print ("Error: empty data!!")
        return 
    #extract features column
    features = main_dict.columns[1:8]
    x_scale = main_dict.columns[8]
    y_scale = main_dict.columns[9]

    # arbitrary threshold
    threshold_isSame = 7
    threshold_isSimilar = 5

    toAppend = pd.DataFrame({
        'isSimilar': [],
        'isSame': []
    })
    for indx, rw in main_dict.iterrows():
        print("\nAnalyzing Face: ", indx, ", ", main_dict['image_name'][indx])
        toAppend.loc[indx] = compareFaces(indx, rw, main_dict, features, x_scale,threshold_isSame, threshold_isSimilar)

    result = pd.concat([main_dict, toAppend], axis=1, sort=False)
    # print (result)
    #store results
    store_csv(result,fileName="../csv_files/final_results.csv")