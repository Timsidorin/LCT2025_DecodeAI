import os


def print_directory_tree(startpath):
    """
    Выводит дерево директории с указанием всех файлов.
    """
    for root, dirs, files in os.walk(startpath):
        # Определяем отступ для текущей директории
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        # Выводим имя директории
        print(f'{indent}{os.path.basename(root)}/')

        # Выводим файлы в этой директории
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{sub_indent}{f}')


# Укажите путь к директории, для которой хотите вывести дерево
# Для текущей директории используйте '.'
start_directory = '.'
print_directory_tree(start_directory)