import imp
from operator import truediv
import os
import profile
from injector import *
from script import *
import time
from dronekit_sitl import SITL
from dronekit import connect, APIException, VehicleMode, LocationGlobalRelative, LocationGlobal, Command
import random
import numpy as np
from datatype import *
from mission1 import *
import sys
from pymavlink import mavutil

from script import recoverAllFiles

bug_id = 0

class SimRunner:
    def arm_vehicle(self):
        arm_time = 0
        self.vehicle.armed = True

        while not self.vehicle.armed:
            time.sleep(0.2)
            print('waiting for armed...')
            arm_time += 0.2
            if arm_time > 100:
                print("Arm failed")
                self.vehicle.close()
                self.sitl.stop()
                self.sitl.p.stdout.close()
                self.sitl.p.stderr.close()
                self.ready = False


    def __init__(self, config, initial_profile):
        ardupilot_dir = config['root_dir'] + 'experiment/elf/'
        out_dir = config['root_dir'] + 'experiment/output/'
        self.elf_dir = "%s%s/" % (ardupilot_dir, "BugFreeTrajectory")
        # the result dictionary
        self.exp_out_dir = "%s%s/%s/" % (out_dir, 'PA', "BugFreeTrajectory")
        self.ready = True
        self.states = []
        self.profiles = []
        copter_file = self.elf_dir + "ArduCopter.elf"
        self.delta = 0.1

        self.sitl = SITL(path = copter_file)

        home_str = "%.6f,%.6f,%.2f,%d" % (initial_profile.lat, initial_profile.lon, initial_profile.alt,
                                          initial_profile.yaw)
        
        sitl_args = ['-S', '-I%d' % bug_id, '--home='+home_str, '--model',
                     '+', '--speedup=1', '--defaults='+self.elf_dir+'copter.parm']
        
        self.sitl.launch(sitl_args, await_ready=True, restart=True, wd=self.elf_dir)
        
        port_number = 5760 + bug_id * 10

        
        try:
            # Use TCP/IP to connect the vehicle on specific port 
            self.vehicle = connect('tcp:127.0.0.1:%d' % port_number, wait_ready=True,rate=10)
        except APIException:
            print("Cannot connect")
            self.sitl.stop()
            self.sitl.p.stdout.close()
            self.sitl.p.stderr.close()
            self.ready = False
            return
        for k, v in initial_profile.params.items():
            self.vehicle.parameters[k] = v

        while not self.vehicle.is_armable:
            print("initializing...")
            time.sleep(1)
        self.arm_vehicle()
        time.sleep(2)
        print(self.vehicle.version)    
            
    def run(self):
        takeoff_mission = Mission1()
        takeoff_mission.run(self)
        time.sleep(10)
        temp_state = []
        waypoint_num = 3
        
        # Fly 4 times. Each time has 3 waypoints. For each waypoint, the vehicle runs for 2 seconds 
        T = 2
        for i in range(0,4):
            current_location = self.vehicle.location.global_frame
            if i % 2 == 0:
                # 0, 2
                target_delta = [random.uniform(0.0002, 0.0003)/waypoint_num, random.uniform(0.0002, 0.0003)/waypoint_num, random.uniform(20, 30)/waypoint_num]
            else:
                # 1, 3
                target_delta = [random.uniform(0.0002, 0.0003)/waypoint_num, random.uniform(-0.0003, -0.0002)/waypoint_num, random.uniform(-30, -20)/waypoint_num]
            # count = 0
            for j in range(1, waypoint_num+1):
                # each time, the move distance will increase. waypoint0 to waypoint1 moves 1 delta, 
                # waypoint1 to waypoint2 moves 2 delta, waypoint2 to waypoint3 moves 3delta
                profile = LocationGlobal(current_location.lat+target_delta[0]*j, current_location.lon+target_delta[1]*j, current_location.alt+target_delta[2]*j)
                self.profiles.append([profile.lat,profile.lon,profile.alt])
                # first to waypoint1, and then waypoint2, then waypoint3
                self.vehicle.simple_goto(profile)
                current_t = 0
                temp_state = []

                # record the [lat,lon,alt,pitch,yaw,roll,velocity] every 0.1 seconds
                # T = 2, the vehicle runs for 2 seconds to get to each waypoint
                while current_t < T:
                    temp_state.append([self.vehicle.location.global_frame.lat, self.vehicle.location.global_frame.lon, self.vehicle.location.global_frame.alt
                    , self.vehicle.attitude.pitch, self.vehicle.attitude.yaw, self.vehicle.attitude.roll, self.vehicle.velocity[0], self.vehicle.velocity[1], self.vehicle.velocity[2]])
                    # currently delta is 0.1s
                    time.sleep(self.delta)
                    current_t += self.delta
                self.states.append(temp_state)

        self.vehicle.close()
        self.sitl.stop()

        np.save(self.exp_out_dir + "states_np_Bug_Free", np.array(self.states))
        np.save(self.exp_out_dir + "profiles_np_Bug_Free", np.array(self.profiles))

        print("Output Execution Path...")
        with open(self.exp_out_dir + "raw_Bug_Free.txt", "w") as execution_file:
            ep_line = self.sitl.stdout.readline(0.01)
            while ep_line is not None:
                execution_file.write(ep_line)
                ep_line = self.sitl.stdout.readline(0.01)

        self.sitl.p.stdout.close()
        self.sitl.p.stderr.close()
        print("Simulation Bug Free completed.")

def run_bug_free():
    config = {}
    config['root_dir'] = "/home/csszhang/dir/arduPilot/"
    for bug in bug_group:
        # for example, recover '/home/ian/Desktop/dir/ArduPilotTestbedForOracleResearch/ardupilot_snapshot/arduPilot/libraries/AC_AttitudeControl/AC_PosControl.cpp' with the file '/home/ian/Desktop/dir/ArduPilotTestbedForOracleResearch/ardupilot_snapshot/arduPilot/.ArduPilot_Back/libraries/AC_AttitudeControl/AC_PosControl.cpp'
        os.system('cp ' + config['root_dir'] + '.ArduPilot_Back/' + bug['file'] + ' ' + config['root_dir'] + bug['file'])
    for bug in real_life_bug_group:
        os.system('cp ' + config['root_dir'] + '.ArduPilot_Back/' + bug['file'] + ' ' + config['root_dir'] + bug['file'])
    
    os.chdir(config['root_dir'])
    os.system('make sitl -j4')
    time.sleep(15)
    os.system('cp build/sitl/bin/arducopter experiment/elf/BugFreeTrajectory/ArduCopter.elf')
    time.sleep(3)
    
    profiles_generated = generate_profiles()
    for core_cnt, profile in enumerate(profiles_generated):
        sim = SimRunner(config, profile)
        if sim.ready:
            ### run four basic guided missions
            sim.run()
        else:
            print("Not ready for mission")
            break
    if sim.ready:
        sim.run()


if __name__ == "__main__":
    run_bug_free()