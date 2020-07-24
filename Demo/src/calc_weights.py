from auxilary import *


def calc_weights():
    '''
        TODO:
            + Find differences between all the combinations of each person's images
                (person 1 has image 1,2,3) -> diff_1_2, diff_1_3, diff_2_3
                if 4 images -> diff_1_2, diff_1_3, diff_1_4, diff_2_3, diff_2_4, diff_3_4

            + store each set of images in a row in the result dataframe

            + check on the range of values (manually, or using some code)

            + convert the values into the largest range (not manually of course)

            + add these values

            + normaliz them -> (value[0] = value[0] / sum of all)

            + weights = 1 - normalized_version (remember the lower the better)

            + weights *= 100

            + done :D
    '''
    print ("hi maggie")
    data = read_csv(fileName=path_to_csv_key_points)
    print (data)
    
    
    
