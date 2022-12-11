import os
import random


target_statements = [
    '_limited_speed_xy_cms = MIN(_limited_speed_xy_cms, get_slow_down_speed(dist_to_dest, _track_accel));',#0
    '_track_desired = track_desired_max;',#1
    'if (_limited_speed_xy_cms < 0.0f) {',#2
    'if (_limited_speed_xy_cms < 0.0f) {',#3
    'if (_flags.fast_waypoint) {',#4
    

]


mutated_statements = [
    '/* BUG!! */ _limited_speed_xy_cms = MAX(_limited_speed_xy_cms, get_slow_down_speed(dist_to_dest, _track_accel));',#0
    '/* BUG!! */ //_track_desired = track_desired_max;',#1
    '/* BUG!! */ if (False) {',#2
    '/* BUG!! */ if (True) {',#3
    '/* BUG!! */ if (False) {',#4

    
]


# Insert EXECUTE_MARK() lables in C++ source code, 
# Once the program passes through the block of code that contains the tag,
# EXECUTE_MARK() will save the file name & function name & line number into code trace files.
# Here the 'lineno' should indicate the typical line number that represent the revelant bug

# target_statements: set of original program statements to be replaced depending on the injected bug(s) chosen.
# mutated_statements: set of buggy program statements to replace depending on the injected bug(s) chosen.
# 'indices': Follow this(these) number(s) to find the program statement(s) in target_statements & mutated_statements.
# 'lineno': When labeling ground truth, find this(these) number(s) in the executation trace(s). 
#           A match indicates the corresponding bug is encountered in the executation.
bug_group = [
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [0],
        "lineno": [412],
    }, # bug0

    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [1],
        "lineno": [427],
    }, # bug1

    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [2],
        "lineno": [427],
    }, # bug2

    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [3],
        "lineno": [427],
    }, # bug2

    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [4],
        "lineno": [451],
    }, # bug4


   
   
]

# bug7 & bug8 can not appear at the same time; nighter can bug17 & bug18
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