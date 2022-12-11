import os
import random


target_statements = [
    # AC_PosControl.cpp
    'vel_max_xy_cms = vel_max_cms * dest_vector_xy_length;', #0
    '_jerk_max_xy_cmsss = MIN(_jerk_max_xy_cmsss, jerk_max_cmsss);', #1
    '_jerk_max_z_cmsss = MIN(_jerk_max_z_cmsss, MIN(GRAVITY_MSS * 100.0, _accel_max_z_cmss) * (M_2PI * _pid_accel_z.filt_T_hz()) / 5.0);', #2
    'thr_out = _pid_accel_z.update_all(_accel_target.z, z_accel_meas, (_motors.limit.throttle_lower || _motors.limit.throttle_upper)) * 0.001f;', #3
    'lean_angles_to_accel_xy(accel_x, accel_y);', #4
    'turn_rate = -turn_rate;}', #5
    'if ((accel_turn.y * vel_desired_xy.x - accel_turn.x * vel_desired_xy.y) < 0.0) { EXECUTE_MARK();', #6
    'turn_rate = -turn_rate;}', #7
    'turn_rate = -turn_rate;}', #8
    '_pos_target.z = curr_pos.z + _p_pos_z.get_error();' #9
]


mutated_statements = [
    # AC_PosControl.cpp
    '/* BUG!! */ //vel_max_xy_cms = vel_max_cms * dest_vector_xy_length;', #0
    '/* BUG!! */ _jerk_max_xy_cmsss = MAX(_jerk_max_xy_cmsss, jerk_max_cmsss);', #1
    '/* BUG!! */ _jerk_max_z_cmsss = MIN(_jerk_max_z_cmsss, MIN(GRAVITY_MSS * 100.0, _accel_max_z_cmss) * (M_2PI * _pid_accel_z.filt_T_hz()) * 5.0);', #2
    '/* BUG!! */ thr_out = _pid_accel_z.update_all(_accel_target.z, _accel_target.z, (_motors.limit.throttle_lower || _motors.limit.throttle_upper)) * 0.001f;', #3
    '/* BUG!! */ //lean_angles_to_accel_xy(accel_x, accel_y);', #4
    '/* BUG!! */ }//turn_rate = -turn_rate;}', #5
    '/* BUG!! */ //if ((accel_turn.y * vel_desired_xy.x - accel_turn.x * vel_desired_xy.y) < 0.0) { EXECUTE_MARK();', #6
    '/* BUG!! */ //turn_rate = -turn_rate;}', #7
    '/* BUG!! */ turn_rate = -turn_rate;//}', #8
    '/* BUG!! */ _pos_target.z = curr_pos.z - _p_pos_z.get_error();' #9
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
    # bugs in AC_PosControl.cpp
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [0],
        "lineno" : [362],
     }, #0
     {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [1],
        "lineno" : [428],
     }, #1
     {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [2],
        "lineno" : [703],
     }, #2
     {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [3],
        "lineno" : [991],
     }, #3
     {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [4],
        "lineno" : [1166],
     }, #4
     {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [5],
        "lineno" : [1245],
     }, #5
     {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [6,7],
        "lineno" : [1240],
     }, #6
     {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [6,8],
        "lineno" : [1240],
     }, #7
     {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [9],
        "lineno" : [1310],
     }, #8

]
