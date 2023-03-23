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
    '', #17
    '', #18
    '', #19
    '_speed_cms = speed_cms;', #20
    '_pos_target.z = curr_alt + _leash_up_z;', #21
    '_pos_error.x = _pos_target.x - curr_pos.x;', #22
    '_pos_error.y = _pos_target.y - curr_pos.y;', #23
    '_accel_cms = accel_cmss;', #24
    '_track_accel = 0;', #25
    '_track_speed = 0;', #26
    '_pos_delta_unit.x = 0;', #27
    '_slow_down_dist = 0.0f;', #28
    'Vector3f dist_to_dest = (curr_pos - Vector3f(0,0,terr_offset)) - _destination;', #29
    '_track_leash_length = leash_z/pos_delta_unit_z;', #30
    '_limited_speed_xy_cms = MIN(_limited_speed_xy_cms,', #31
    '_limited_speed_xy_cms -= 2.0f * _track_accel * dt;', #32
    '_limited_speed_xy_cms = 0;', #33
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
    '', #17
    '', #18
    '', #19
    '/* BUG!! */ _speed_cms = -speed_cms;', #20
    '/* BUG!! */ _pos_target.z = curr_alt - _leash_up_z;', #21
    '/* BUG!! */ _pos_error.x = curr_pos.x - _pos_target.x;', #22
    '/* BUG!! */ _pos_error.y = curr_pos.y - _pos_target.y;', #23
    '/* BUG!! */ _accel_cms = -accel_cmss;', #24
    '/* BUG!! */ //_track_accel = 0;', #25
    '/* BUG!! */ //_track_speed = 0;', #26
    '/* BUG!! */ return false;', #27
    '/* BUG!! */ //_slow_down_dist = 0.0f;', #28
    '/* BUG!! */ Vector3f dist_to_dest = (curr_pos + Vector3f(0,0,terr_offset)) - _destination;', #29
    '/* BUG!! */ //_track_leash_length = leash_z/pos_delta_unit_z;', #30
    '/* BUG!! */ _limited_speed_xy_cms = MAX(_limited_speed_xy_cms,', #31
    '/* BUG!! */ _limited_speed_xy_cms += 2.0f * _track_accel * dt;', #32
    '/* BUG!! */ //_limited_speed_xy_cms = 0;', #33
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

    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [20],
        "lineno" : [654], 
    }, #bug15 V2 bug_group #1

    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [21],
        "lineno" : [517],
    }, #bug16 V2 bug_group #3

    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [22, 23],
        "lineno" : [1009],
    }, #bug17 V2 bug_group #4

    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [24],
        "lineno" : [644],
    }, #bug18 V2 bug_group #5

    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [25,26],
        "lineno" : [558],
    }, #19 Y 558 100%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [27],
        "lineno" : [250],
    }, #20 Y 250
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [28],
        "lineno" : [1017],
    }, #21 Y 1017 100%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [29],
        "lineno" : [455],
    }, #22 Y 455 49.4%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [30],
        "lineno" : [566],
    }, #23 Y 566 62.4%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [31],
        "lineno" : [412],
    }, #24 Y 412 51.8%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [32],
        "lineno" : [427],
    }, #25 Y 427 98%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [33],
        "lineno" : [394],
    }, #26 Y 394 6.4%
]



real_life_target_statements = [
    'float acro_level_mix = constrain_float(1-float(MAX(MAX(abs(roll_in), abs(pitch_in)), abs(yaw_in))/4500.0), 0, 1)*ahrs.cos_pitch();', #0
    'leash_length = POSCONTROL_LEASH_LENGTH_MIN', #1
    'loiter_nav->clear_pilot_desired_acceleration();', #2
    'accel_x_cmss = (GRAVITY_MSS * 100) * (-_ahrs.cos_yaw() * _ahrs.sin_pitch() * _ahrs.cos_roll() - _ahrs.sin_yaw() * _ahrs.sin_roll()) / MAX(_ahrs.cos_roll()*_ahrs.cos_pitch(), 0.5f);', #3
    'accel_y_cmss = (GRAVITY_MSS * 100) * (-_ahrs.sin_yaw() * _ahrs.sin_pitch() * _ahrs.cos_roll() + _ahrs.cos_yaw() * _ahrs.sin_roll()) / MAX(_ahrs.cos_roll()*_ahrs.cos_pitch(), 0.5f);', #4
    '', #5
    'return MAX(ToDeg(_althold_lean_angle_max), AC_ATTITUDE_CONTROL_ANGLE_LIMIT_MIN) * 100.0f;', #6
    'if ((vector_length > max_length) && is_positive(vector_length))', #7
    'vector_x *= (max_length / vector_length);', #8
    'vector_y *= (max_length / vector_length);', #9
    '_slow_down_dist = speed_cms * speed_cms / (4.0f*accel_cmss);', #10
]

real_life_mutated_statements = [
    '/*!! BUG !! */ float acro_level_mix = constrain_float(float(1-MAX(MAX(abs(roll_in), abs(pitch_in)), abs(yaw_in))/4500.0), 0, 1)*ahrs.cos_pitch();', #0
    '/* BUG!! */ //leash_length = POSCONTROL_LEASH_LENGTH_MIN', #1
    '/*!! BUG !! */ //loiter_nav->clear_pilot_desired_acceleration();', #2
    '/* BUG!! */ accel_x_cmss = (GRAVITY_MSS * 100) * (-(_ahrs.cos_yaw() * _ahrs.sin_pitch() / MAX(_ahrs.cos_pitch(),0.5f)) - _ahrs.sin_yaw() * _ahrs.sin_roll() / MAX(_ahrs.cos_roll(),0.5f));', #3
    '/* BUG!! */ accel_y_cmss = (GRAVITY_MSS * 100) * (-(_ahrs.sin_yaw() * _ahrs.sin_pitch() / MAX(_ahrs.cos_pitch(),0.5f)) + _ahrs.cos_yaw() * _ahrs.sin_roll() / MAX(_ahrs.cos_roll(),0.5f));', #4
    '', #5
    '/*!! BUG !! */ return ToDeg(_althold_lean_angle_max) * 100.0f;', #6
    '/*!! BUG !! */ if ((vector_length > max_length) && is_positive(max_length))', #7
    '/* BUG!! */ vector_x = (max_length / vector_length);', #8
    '/* BUG!! */ vector_y = (max_length / vector_length);', #9
    '/* BUG!! */ _slow_down_dist = speed_cms * speed_cms / (2.0f*accel_cmss);',#10
]

real_life_bug_group = [
    {
        'file':'ArduCopter/mode_acro.cpp',
        'indices':[0],
        'lineno':[140]
    }, #0
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [1],
        "lineno" : [1148],
    }, #1 22.7%
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [3,4],
        "lineno" : [1116],
    }, #2
    {
        'file':'libraries/AC_AttitudeControl/AC_AttitudeControl.cpp',
        'indices':[6],
        'lineno':[913]
    }, #3 
    {
        'file':'libraries/AC_AttitudeControl/AC_PosControl.cpp',
        'indices':[7],
        'lineno':[1198]
    }, #4 
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [8,9],
        "lineno" : [1198],
    }, #5 10.1%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [10],
        "lineno" : [1015],
    }, #6 100%
]

'''
potential realife bugs:
pull request #8191;  on main branches & too many lines
#10388: on main branches & not copter
#9889: future bugs
#7273: on main branch
'''


old_bug_group = [
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [0],
        "lineno" : [430],
     }, #0
     {
         "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [1],
        "lineno" : [654],
    }, #1 Y 654 100%
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [2,3],
        "lineno" : [427,432],
    }, #2
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [4],
        "lineno" : [517],
    }, #3 Y 517 100%
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [5,6],
        "lineno" : [1009],
    }, #4 Y 1009 100%
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [7],
        "lineno" : [644],
    }, #5 Y 644 100%
    {
        "file":"libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices":[8],
        'lineno':[287]
    }, #6 
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [21,22],
        "lineno" : [558],
    }, #7 Y 558 100%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [23],
        "lineno" : [250],
    }, #8 Y 250
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [24],
        "lineno" : [1017],
    }, #9 Y 1017 100%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [25],
        "lineno" : [455],
    }, #10 Y 455 49.4%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [26],
        "lineno" : [566],
    }, #11 Y 566 62.4%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [27],
        "lineno" : [412],
    }, #12 Y 412 51.8%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [28],
        "lineno" : [427],
    }, #13 Y 427 98%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [29],
        "lineno" : [394],
    }, #14 Y 394 6.4%
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [30,31,32],
        "lineno" : [545,549,553],
    }, #15 
]



def check_injection_validation(bug_id_list, config):
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

    group = []
    t_statements = []
    m_statements = []
    if config['real_life'] == 'True':
        group = real_life_bug_group
        t_statements = real_life_target_statements
        m_statements = real_life_mutated_statements
    else:
        group = bug_group
        t_statements = target_statements
        m_statements = mutated_statements

    for id in selected_bugs_id:
        print(id)
        selected_bugs.append(group[id])

    for bug in selected_bugs:
        relative_file = open(config['root_dir'] + bug['file'])
        relative_file_data = relative_file.read()
        relative_file.close()
        for index in bug['indices']:
            if t_statements[index] not in relative_file_data:
                print(t_statements[index] + 'not in' + bug['file'])
                exit(-1)
            relative_file_data = relative_file_data.replace(t_statements[index], m_statements[index])
        new_file = '%s%s' % (dest_dir, bug['file'])
        if not os.path.exists(os.path.dirname(new_file)):
            os.makedirs(os.path.dirname(new_file))
        with open(new_file, 'w') as f:
            f.write(relative_file_data)