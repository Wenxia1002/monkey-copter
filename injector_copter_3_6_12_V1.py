import os
import random
from itertools import combinations


target_statements = [
    '_limited_speed_xy_cms = MIN(_limited_speed_xy_cms, get_slow_down_speed(dist_to_dest, _track_accel));',#0
    '_track_desired = track_desired_max;',#1
    'if (_limited_speed_xy_cms < 0.0f) {',#2
    'if (_limited_speed_xy_cms < 0.0f) {',#3
    'if (_flags.fast_waypoint) {',#4
    '_limited_speed_xy_cms = MIN(_limited_speed_xy_cms, get_slow_down_speed(dist_to_dest, _track_accel));', #5
    '_limited_speed_xy_cms = MIN(_limited_speed_xy_cms, get_slow_down_speed(dist_to_dest, _track_accel));', #6
    '_track_desired = track_desired_max;', #7
    'Vector3f dist_to_dest = (curr_pos - Vector3f(0,0,terr_offset)) - _destination;', #8
    '_limited_speed_xy_cms = MIN(_limited_speed_xy_cms, get_slow_down_speed(dist_to_dest, _track_accel));', #9
    'stopping_point.z = curr_pos_z + (linear_distance + curr_vel_z*curr_vel_z/(2.0f*_accel_z_cms));', #10
    '_pos_error.z = _leash_up_z;', #11
    '_pos_target.x = curr_pos.x + _pos_error.x;', #12
    '_pos_target.y = curr_pos.y + _pos_error.y;', #13
    'leash_length = POSCONTROL_LEASH_LENGTH_MIN;', #14
    'vector_x *= (max_length / vector_length);', #15
    'vector_y *= (max_length / vector_length);', #16
]


mutated_statements = [
    '/* BUG!! */ _limited_speed_xy_cms = MAX(_limited_speed_xy_cms, get_slow_down_speed(dist_to_dest, _track_accel));',#0
    '/* BUG!! */ //_track_desired = track_desired_max;',#1
    '/* BUG!! */ if (false) {',#2
    '/* BUG!! */ if (true) {',#3
    '/* BUG!! */ if (false) {',#4
    '/* BUG!! */ _limited_speed_xy_cms = MIN(_limited_speed_xy_cms, get_slow_down_speed(dist_to_dest, _track_speed));', #5
    '/* BUG!! */ //_limited_speed_xy_cms = MIN(_limited_speed_xy_cms, get_slow_down_speed(dist_to_dest, _track_accel));', #6
    '/* BUG!! */ //_track_desired = track_desired_max;', #7
    '/* BUG!! */ Vector3f dist_to_dest = Vector3f(0,0,0);', #8
    '/* BUG!! */ _limited_speed_xy_cms = MIN(2.0f * _track_accel * dt, get_slow_down_speed(dist_to_dest, _track_accel));', #9
    '/* BUG!! */ //stopping_point.z = curr_pos_z + (linear_distance + curr_vel_z*curr_vel_z/(2.0f*_accel_z_cms));', #10
    '/* BUG!! */ //_pos_error.z = _leash_up_z;', #11
    '/* BUG!! */ //_pos_target.x = curr_pos.x + _pos_error.x;', #12
    '/* BUG!! */ //_pos_target.y = curr_pos.y + _pos_error.y;', #13
    '/* BUG!! */ leash_length = 200.0f;', #14
    '/* BUG!! */ //vector_x *= (max_length / vector_length);', #15
    '/* BUG!! */ //vector_y *= (max_length / vector_length);', #16
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
    }, # bug3

    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [4],
        "lineno": [451],
    }, # bug4

    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [5],
        "lineno": [412],
    }, # bug5 Wrong Variable used in Parameter of Function call

    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [6],
        "lineno": [412],
    }, # bug6 Missing Functin call

    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [7],
        "lineno": [427],
    }, # bug7 Missing Variable Assignment using a value

    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [8],
        "lineno": [455],
    }, # bug8 Missing Variable Initialization using a value

    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [9],
        "lineno": [412],
    }, # bug9 Wrong Arithmetic Expression used in Parameter of Function call

    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [10],
        "lineno": [437],
    }, # bug10 Missing Variable Assignment using expression 

    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [11],
        "lineno": [517],
    }, # bug11 Wrong Value Assigned to Variable 

    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [12, 13],
        "lineno": [1018],
    }, # bug12 Missing Variable Assignment using expression 

    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [14],
        "lineno": [1148],
    }, # bug13 Wrong Value Assigned to Variable 

    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [15, 16],
        "lineno": [1198],
    }, # bug14  
]


# TODO: test this function
def check_injection_validation(bug_id_list):
    bug_id_list_set = set(bug_id_list)
    conflict_index_list = [[0, 5, 6, 9], [1, 2, 3, 7], [4, 8]]
    conflict_combination_list = []

    for conflict_index in conflict_index_list:
        for length in (2, len(conflict_index)):
            temp_combination = combinations(conflict_index, length)
            for combination in temp_combination:
                conflict_combination_list.append(combination)

    # if one of the combinations is totally contained by the chosen bug_id_list
    for conbination in conflict_combination_list:
        if len(set(conbination) - bug_id_list_set) == 0:
            return False

    return True


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