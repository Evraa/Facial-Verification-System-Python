def rename_datset():
    import os
    path = '../dataset/'
    files = os.listdir(path)
    i = 0
    for img_file in files:
        file_path = path + img_file
        new_name = path + str(i) + '.jpg'
        i+=1
        os.rename(file_path,new_name)


def print_code():
    dominant_key_points = [17, 21, 22, 26, 36, 39, 42, 45, 32, 34, 48, 54]
    code = "my_dict = {'image_set': [], \n"
    line = "'feat_"
    end = "': [],\n"
    for d in dominant_key_points:
        added = line + str(d)
        code += added
        code += end
    print (code)

def print_code_2():
    dominant_key_points = [17, 21, 22, 26, 36, 39, 42, 45, 32, 34, 48, 54]
    code_1 = "my_dict['feat_"
    code_2 = "'].append(list(shape["
    end = "]))"
    for d in dominant_key_points:
        code = code_1
        code += str(d)
        code += code_2
        code += str(d)
        code +=  end
        print (code)


if __name__ == "__main__":
    print_code_2()