import os
import random


target_statements = [
    # AC_PosControl.cpp
    'float posz = pos.z;', #0
    'Vector2f accel_target = _pid_vel_xy.update_all(_vel_target.xy(), _vehicle_horiz_vel, _limit_vector.xy());', #1
    '_pid_accel_z.relax_integrator((throttle_setting - _motors.get_throttle_hover()) * 1000.0f, POSCONTROL_RELAX_TC);', #2
    'update_pos_offset_z(pos_offset_z);', #3
    '_accel_target.x += _accel_desired.x;', #4
    '_accel_target.y += _accel_desired.y;', #5
    '_pos_target.z = pos_target_zf;', #6
    'vel_max_xy_cms = vel_max_cms * dest_vector_xy_length;', #7
    'if (!limit_accel_xy(_vel_desired.xy(), _accel_target.xy(), accel_max)) {', #8
    '_limit_vector.xy().zero();}', #9
    'if (!limit_accel_xy(_vel_desired.xy(), _accel_target.xy(), accel_max)) {', #10
    '_limit_vector.xy().zero();}', #11
    'else if (_motors.limit.throttle_lower) {', #12
    'float angle_max = MIN(_attitude_control.get_althold_lean_angle_max(), get_lean_angle_max_cd());', #13
    'float accel_max = GRAVITY_MSS * 100.0f * tanf(ToRad(angle_max * 0.01f));', #14
    'float dest_vector_xy_length = dest_vector.xy().length();', #15
    'else {  EXECUTE_MARK();', #16
    '_limit_vector.z = 0.0f;}', #17
    '', #18
    '', #19
    # AC_AttitudeControl.cpp
    '_euler_angle_target.z = euler_yaw_angle;', #20
    '_euler_angle_target.z = wrap_PI(angle_error + _euler_angle_target.z);', #21
    'attitude_error.z = constrain_float(wrap_PI(attitude_error.z), -AC_ATTITUDE_ACCEL_Y_CONTROLLER_MAX_RADSS / _p_angle_yaw.kP(), AC_ATTITUDE_ACCEL_Y_CONTROLLER_MAX_RADSS / _p_angle_yaw.kP());', #22
    'ang_vel_limit(ang_vel, radians(_ang_vel_roll_max), radians(_ang_vel_pitch_max), 0.0f);', #23
    '_ang_vel_body.z += ang_vel_body_feedforward.z;', #24
    'target_ang_vel.y = ang_vel.y;', #25
    'attitude_target = attitude_body * thrust_vector_correction * yaw_vec_correction_quat;', #26
    'if (reset_rate ) {', #27
    '_euler_angle_target.zero();}', #28
    'if (reset_rate ) {', #29
    '_ang_vel_target.zero();//', #30
    '_euler_angle_target.zero();}', #31
    'else if (_thrust_error_angle > AC_ATTITUDE_THRUST_ERROR_ANGLE) {', #32
    '_feedforward_scalar = (1.0f - (_thrust_error_angle - AC_ATTITUDE_THRUST_ERROR_ANGLE) / AC_ATTITUDE_THRUST_ERROR_ANGLE);', #33
    '_ang_vel_body.x += ang_vel_body_feedforward.x * _feedforward_scalar;',#34
    '_ang_vel_body.y += ang_vel_body_feedforward.y * _feedforward_scalar; EXECUTE_MARK();', #35
    '_ang_vel_body.z += ang_vel_body_feedforward.z;', #36
    '_ang_vel_body.z = _ahrs.get_gyro().z * (1.0 - _feedforward_scalar) + _ang_vel_body.z * _feedforward_scalar;}', #37
    'float err_mag = attitude_error.length();', #38
    'if (_thrust_error_angle > AC_ATTITUDE_THRUST_ERROR_ANGLE * 2.0f) { EXECUTE_MARK();', #39
    '_ang_vel_body.z = _ahrs.get_gyro().z;}', #40
    'else {   EXECUTE_MARK();', #41
    '_ang_vel_body += ang_vel_body_feedforward;}', #42
    '_ang_vel_body += ang_vel_body_feedforward;}', #43
    'else if (_thrust_error_angle > AC_ATTITUDE_THRUST_ERROR_ANGLE) {', #44
    '', #45
    '', #46
    '', #47
    '', #48
    '', #49
    # AC_WPNav.cpp
    'target_vel *= vel_time_scalar;', #50
    'set_speed_xy(_wp_speed_cms);', #51
    'track_scaler_dt = constrain_float(0.05f + (track_velocity - _pos_control.get_pos_xy_p().kP() * track_error) / curr_target_vel.length(), 0.1f, 1.0f);', #52
    'update_track_with_speed_accel_limits();', #53
    'target_pos.z += _pos_control.get_pos_offset_z_cm();', #54
    '_origin.x = curr_pos.x;', #55
    '_terrain_vel = speed_cms * _terrain_vel / _wp_desired_speed_xy_cms;', #56
    'if (!is_active() || !_flags.reached_destination ) {', #57
    'wp_and_spline_init(_wp_desired_speed_xy_cms);}', #58
    'if (!is_active() || !_flags.reached_destination ) {', #59
    'wp_and_spline_init(_wp_desired_speed_xy_cms);}', #60
    'float origin_speed = 0.0f;', #61
    'if (is_zero(_scurve_jerk)) { EXECUTE_MARK();', #62
    '_scurve_jerk = _wp_jerk;', #63
    '} else  { EXECUTE_MARK();', #64
    '_scurve_jerk = MIN(_scurve_jerk, _wp_jerk); }', #65
]


mutated_statements = [
    # AC_PosControl.cpp
    '/* BUG!! */ float posz = -pos.z;', #0
    '/* BUG!! */ Vector2f accel_target = _pid_vel_xy.update_all(_vehicle_horiz_vel, _vel_target.xy(), _limit_vector.xy());', #1
    '/* BUG!! */ _pid_accel_z.relax_integrator((throttle_setting + _motors.get_throttle_hover()) * 1000.0f, POSCONTROL_RELAX_TC);', #2
    '/* BUG!! */ //update_pos_offset_z(pos_offset_z);', #3
    '/* BUG!! */ //_accel_target.x += _accel_desired.x;', #4
    '/* BUG!! */ //_accel_target.y += _accel_desired.y;', #5
    '/* BUG!! */ //_pos_target.z = pos_target_zf;', #6
    '/* BUG!! */ //vel_max_xy_cms = vel_max_cms * dest_vector_xy_length;', #7
    '/* BUG!! */ //if (!limit_accel_xy(_vel_desired.xy(), _accel_target.xy(), accel_max)) { EXECUTE_MARK();', #8
    '/* BUG!! */ _limit_vector.xy().zero(); EXECUTE_MARK();', #9
    '/* BUG!! */ EXECUTE_MARK(); //if (!limit_accel_xy(_vel_desired.xy(), _accel_target.xy(), accel_max)) { EXECUTE_MARK();', #10
    '/* BUG!! */ //_limit_vector.xy().zero();}', #11
    '/* BUG!! */ else {   EXECUTE_MARK();', #12
    '//float angle_max = MIN(_attitude_control.get_althold_lean_angle_max(), get_lean_angle_max_cd());', #13
    '// float accel_max = GRAVITY_MSS * 100.0f * tanf(ToRad(angle_max * 0.01f));', #14
    '// float dest_vector_xy_length = dest_vector.xy().length();', #15
    '//else {  EXECUTE_MARK();', #16
    '//_limit_vector.z = 0.0f;}', #17
    '', #18
    '', #19
    # AC_AttitudeControl.cpp
    '/* BUG!! */ _euler_angle_target.z = 0.0;', #20
    '/* BUG!! */ _euler_angle_target.z = wrap_PI(angle_error + _euler_angle_target.y);', #21
    '/* BUG!! */ attitude_error.z = constrain_float(wrap_PI(attitude_error.z), -AC_ATTITUDE_ACCEL_Y_CONTROLLER_MAX_RADSS * _p_angle_yaw.kP(), AC_ATTITUDE_ACCEL_Y_CONTROLLER_MAX_RADSS * _p_angle_yaw.kP());', #22
    '/* BUG!! */ //ang_vel_limit(ang_vel, radians(_ang_vel_roll_max), radians(_ang_vel_pitch_max), 0.0f);', #23
    '/* BUG!! */ //_ang_vel_body.z += ang_vel_body_feedforward.z;', #24
    '/* BUG!! */ //target_ang_vel.y = ang_vel.y;', #25
    '/* BUG!! */ //attitude_target = attitude_body * thrust_vector_correction * yaw_vec_correction_quat;', #26
    '/* BUG!! */ EXECUTE_MARK(); //if (reset_rate ) {', #27
    '/* BUG!! */ _euler_angle_target.zero();', #28
    '/* BUG!! */ //if (reset_rate ) {', #29
    '/* BUG!! */ //_ang_vel_target.zero();//', #30
    '/* BUG!! */ EXECUTE_MARK(); //_euler_angle_target.zero();}', #31
    'else { EXECUTE_MARK();', #32
    '/* BUG!! */ //_feedforward_scalar = (1.0f - (_thrust_error_angle - AC_ATTITUDE_THRUST_ERROR_ANGLE) / AC_ATTITUDE_THRUST_ERROR_ANGLE);', #33
    '/* BUG!! */ //_ang_vel_body.x += ang_vel_body_feedforward.x * _feedforward_scalar;',#34
    '/* BUG!! */ //_ang_vel_body.y += ang_vel_body_feedforward.y * _feedforward_scalar; EXECUTE_MARK();', #35
    '/* BUG!! */ //_ang_vel_body.z += ang_vel_body_feedforward.z;', #36
    '/* BUG!! */ //_ang_vel_body.z = _ahrs.get_gyro().z * (1.0 - _feedforward_scalar) + _ang_vel_body.z * _feedforward_scalar;}', #37
    '/* BUG!! */ float err_mag;', #38
    '/* BUG!! */ //if (_thrust_error_angle > AC_ATTITUDE_THRUST_ERROR_ANGLE * 2.0f) { EXECUTE_MARK();', #39
    '/* BUG!! */ //_ang_vel_body.z = _ahrs.get_gyro().z;}', #40
    '/* BUG!! */ //else {   EXECUTE_MARK();', #41
    '/* BUG!! */ EXECUTE_MARK(); _ang_vel_body += ang_vel_body_feedforward;', #42
    '//_ang_vel_body += ang_vel_body_feedforward;}', #43
    '// else if (_thrust_error_angle > AC_ATTITUDE_THRUST_ERROR_ANGLE) {', #44
    '', #45
    '', #46
    '', #47
    '', #48
    '', #49
    # AC_WPNav.cpp
    'target_vel = target_vel;', #50
    'set_speed_xy(_last_wp_speed_cms);', #51
    'track_scaler_dt = constrain_float(0.05f + (track_velocity + _pos_control.get_pos_xy_p().kP() * track_error) / curr_target_vel.length(), 0.1f, 1.0f);', #52
    '// update_track_with_speed_accel_limits();', #53
    '// target_pos.z += _pos_control.get_pos_offset_z_cm();', #54
    '// _origin.x = curr_pos.x;', #55
    '// _terrain_vel = speed_cms * _terrain_vel / _wp_desired_speed_xy_cms;', #56
    'EXECUTE_MARK(); //if (!is_active() || !_flags.reached_destination ) {', #57
    'wp_and_spline_init(_wp_desired_speed_xy_cms);', #58
    '// if (!is_active() || !_flags.reached_destination ) {', #59
    'EXECUTE_MARK(); // wp_and_spline_init(_wp_desired_speed_xy_cms);}', #60
    'float origin_speed;', #61
    '// if (is_zero(_scurve_jerk)) { EXECUTE_MARK();', #62
    '// _scurve_jerk = _wp_jerk;', #63
    '// } else  { EXECUTE_MARK();', #64
    'EXECUTE_MARK(); _scurve_jerk = MIN(_scurve_jerk, _wp_jerk);', #65
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
        "lineno": [378],
    }, # bug0
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [1],
        "lineno": [645],
    }, # bug1
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [2],
        "lineno": [767],
    }, # bug2
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [3],
        "lineno": [386],
    }, # bug3
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [4, 5],
        "lineno": [654],
    }, # bug4
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [6],
        "lineno": [964],
    }, # bug5
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [7, 15],
        "lineno": [366],
    }, # bug6
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [8, 9, 13, 14],
        "lineno": [667],
    }, # bug7
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [10, 11, 13, 14],
        "lineno": [665],
    }, # bug8
    {
        "file": "libraries/AC_AttitudeControl/AC_PosControl.cpp",
        "indices": [12, 16, 17],
        "lineno": [1012],
    }, # bug9

# bugs in AC_AttitudeControl.cpp
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [20],
        "lineno": [345],
    }, # bug10
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [21],
        "lineno": [343],
    }, # bug11
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [22],
        "lineno": [733],
    }, # bug12
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [23],
        "lineno": [831],
    }, # bug13
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [24],
        "lineno": [698],
    }, # bug14
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [25],
        "lineno": [834],
    }, # bug15
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [26],
        "lineno": [735],
    }, # bug16
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [27, 28],
        "lineno": [888],
    }, # bug17
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [29, 30, 31],
        "lineno": [891],
    }, # bug18
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [32, 41, 43],
        "lineno": [695],
    }, # bug19
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [38],
        "lineno": [475],
    }, # bug20  Does not work
    {
        "file": "libraries/AC_AttitudeControl/AC_AttitudeControl.cpp",
        "indices": [33, 34, 35, 36, 37, 39, 40, 41, 42, 44],
        "lineno": [702],
    }, # bug21
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [50],
        "lineno": [506],
    }, # bug22
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [51],
        "lineno": [577],
    }, # bug23
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [52],
        "lineno": [472],
    }, # bug24
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [53],
        "lineno": [215],
    }, # bug25
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [54],
        "lineno": [510],
    }, # bug26
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [55],
        "lineno": [485],
    }, # bug27
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [56],
        "lineno": [205],
    }, # bug28
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [57, 58],
        "lineno": [284],
    }, # bug29
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [59, 60],
        "lineno": [285],
    }, # bug30
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [61],
        "lineno": [288],
    }, # bug31 Does not work
    {
        "file": "libraries/AC_WPNav/AC_WPNav.cpp",
        "indices": [62, 63, 64, 65],
        "lineno": [859],
    }, # bug32
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