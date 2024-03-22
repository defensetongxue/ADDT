import os
dataset_path = 'E:/CS dataset/Zhongshan Dataset 2023.3.19-1.12 combine/1.Zhongshan data combine1.12-3.19/dataset_anomaly_detection_WholeFace'
# dataset_path = 'E:/CS dataset/Zhongshan Dataset 2023.3.19-1.12 combine/1.Zhongshan data combine1.12-3.19/dataset_anomaly_detection_WholeFace/test'
# dataset_path = '/dataset/dataset_contrast_module_train_test/test' # 这是用于放在云脑上训练的数据集text list
txt_path = 'D:/CS Phd/2.中山大学眼科医院合作/4.异常检测\异常检测/3.My_distribution_change_method'
# 打开文件，以写入模式打开，并将其保存到变量file中
dataset_type = "test"     # 从这里修改数据类型，写train or test 
dataset_path = dataset_path + "/" + dataset_type
prefix = "/datalist_"
suffix = ".txt"
txt_path = txt_path + f"{prefix}{dataset_type}{suffix}"

def numerical_sort(value):
    # 获取文件名中的数字部分
    num = int(''.join(filter(str.isdigit, value)))
    return num

with open(txt_path, 'w') as file:
    if dataset_type == "train":
        # 在文件中写入文本，并添加换行符
        # 路径
        good_path = dataset_path + "/good"
        base_path = dataset_path + "/base"  # 参考的图片
        # 读取文件夹，并且按顺序排列
        visible_person_files = os.listdir(good_path)
        visible_person_files.sort(key = numerical_sort)

        # visible文件的
        for person in visible_person_files:
            person_path = good_path + f"/{person}"
            base_person_path = base_path + f"/{person}"
            # 一个人的图片名字列表
            image_name_list = os.listdir(person_path)
            image_name_list.sort(key=numerical_sort)
            base_image_names = os.listdir(base_person_path)
            for base_image_name in base_image_names:
                base_person_image_path = base_person_path + "/" + base_image_name
                # 使用os.path.relpath函数获取相对路径
                relative_base_person_image_path = os.path.relpath(base_person_image_path, dataset_path)
                relative_base_person_image_path = dataset_type + "/" + relative_base_person_image_path
                print(relative_base_person_image_path)
            for image_name in image_name_list:
                image_path = person_path + "/" + image_name
                # 使用os.path.relpath函数获取相对路径
                relative_image_path = os.path.relpath(image_path, dataset_path)
                relative_image_path = dataset_type + "/" + relative_image_path
                # print(image_path)
                # visible是正样本，其标签是0
                file.write(relative_image_path + " _0" + " _" + relative_base_person_image_path + "\n")

                # file.write('这是第一行文本。\n这是第二行文本。\n这是第三行文本。')    

    else:
        # 在文件中写入文本，并添加换行符
        # 路径
        good_path = dataset_path + "/good"    
        base_path = dataset_path + "/base"  # 参考的图片
        # 读取文件夹，并且按顺序排列
        visible_person_files = os.listdir(good_path)
        visible_person_files.sort(key = numerical_sort)

        # 在train中没有这个，而test中有
        invisible_path = dataset_path + "/invisible"  
        invisible_person_files = os.listdir(invisible_path)
        invisible_person_files.sort(key = numerical_sort)

        # visible文件的
        for person in visible_person_files:
            person_path = good_path + f"/{person}"
            base_person_path = base_path + f"/{person}"
            # 一个人的图片名字列表
            image_name_list = os.listdir(person_path)
            image_name_list.sort(key=numerical_sort)
            base_image_names = os.listdir(base_person_path)
            for base_image_name in base_image_names:
                base_person_image_path = base_person_path + "/" + base_image_name
                # 使用os.path.relpath函数获取相对路径
                relative_base_person_image_path = os.path.relpath(base_person_image_path, dataset_path)
                relative_base_person_image_path = dataset_type + "/" + relative_base_person_image_path
                print(relative_base_person_image_path)
            for image_name in image_name_list:
                image_path = person_path + "/" + image_name
                # 使用os.path.relpath函数获取相对路径
                relative_image_path = os.path.relpath(image_path, dataset_path)
                relative_image_path = dataset_type + "/" + relative_image_path
                # print(image_path)
                # visible是正样本，其标签是0
                file.write(relative_image_path + " _0" + " _" + relative_base_person_image_path + "\n")

                # file.write('这是第一行文本。\n这是第二行文本。\n这是第三行文本。')    

        # invisible文件的，在train中没有这个，而test中有
        for person in invisible_person_files:
            person_path = invisible_path + f"/{person}"
            base_person_path = base_path + f"/{person}"
            # 一个人的图片名字列表
            image_name_list = os.listdir(person_path)
            image_name_list.sort(key=numerical_sort)
            base_image_names = os.listdir(base_person_path)
            for base_image_name in base_image_names:
                base_person_image_path = base_person_path + "/" + base_image_name
                relative_base_person_image_path = os.path.relpath(base_person_image_path, dataset_path)
                relative_base_person_image_path = dataset_type + "/" + relative_base_person_image_path
                
                print(relative_base_person_image_path)
            for image_name in image_name_list:
                image_path = person_path + "/" + image_name
                # 使用os.path.relpath函数获取相对路径
                relative_image_path = os.path.relpath(image_path, dataset_path)
                relative_image_path = dataset_type + "/" + relative_image_path
                # print(image_path)
                # invisible代表负样本，其标签是1
                file.write(relative_image_path + " _1" + " _" + relative_base_person_image_path + "\n")


