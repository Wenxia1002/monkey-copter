import os
import random

target_statements = [
    '_flags.reached_destination =  true;',
]

mutated_statements = [
    '_flags.reached_destination =  false;',
]

bug_group = [
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [0],
        "lineno": [458],
    },
]

def inject_bugs(selected_bugs_id, config):
    dest_dir = config['root_dir']
    selected_bugs = []

    for id in selected_bugs_id:
        print(id)
        selected_bugs.append(bug_group[id])

    for bug in selected_bugs:
        relative_file = open(config['root_dir'] + bug['file'])
        relative_file_data = relative_file.read()
        relative_file.close()
        for index in bug['indices']:
            if target_statements[index] not in relative_file_data:
                print(target_statements[index] + 'not in' + bug['file'])
                exit(-1)
            relative_file_data = relative_file_data.replace(target_statements[index], mutated_statements[index])
        new_file = '%s%s' % (dest_dir, bug['file'])
        if not os.path.exists(os.path.dirname(new_file)):
            os.makedirs(os.path.dirname(new_file))
        with open(new_file, 'w') as f:
            f.write(relative_file_data)