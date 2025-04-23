#!/usr/bin/env python3

import os
import argparse

def replace_spaces_with_underscores_in_folder_names(folder_path):
    """
    遍历指定文件夹及其所有子文件夹，将文件夹名称中的空格替换为下划线
    :param folder_path: str, 需要处理的根文件夹路径
    """
    for root, dirs, _ in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            original_dir_path = os.path.join(root, dir_name)
            new_dir_name = dir_name.replace(' ', '_')
            new_dir_path = os.path.join(root, new_dir_name)

            if original_dir_path != new_dir_path:
                os.rename(original_dir_path, new_dir_path)
                print(f"Renamed: '{original_dir_path}' -> '{new_dir_path}'")

def main():
    parser = argparse.ArgumentParser(description="Replace spaces in folder names with underscores.")
    parser.add_argument("-d", "--directory", type=str, required=True, help="The root directory to process.")
    args = parser.parse_args()

    folder_path = args.directory
    if os.path.exists(folder_path):
        replace_spaces_with_underscores_in_folder_names(folder_path)
        print("Folder renaming completed!")
    else:
        print("The specified folder path does not exist.")

if __name__ == "__main__":
    main()
