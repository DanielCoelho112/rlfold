import carla
import numpy as np

from utilities.conversions import convert_speed_01, convert_01, convert_11, step_function

def carla_control(action):
    control = carla.VehicleControl(
        throttle= float(action[0]), steer=float(action[1]), brake=float(action[2]))
    return control


   
 
class PID():
    def __init__(self, kp, ki, kd, dt, maximum_speed):
        self.kp = kp 
        self.ki = ki 
        self.kd = kd 
        self.dt = dt
        self.e = 0
        self.e_prev = 0
        self.maximum_speed = maximum_speed
    
    def get(self, action, velocity):
        # desired_velocity = action[0] * self.maximum_speed
        desired_velocity = convert_speed_01(action[0])
        velocity = velocity / self.maximum_speed
        steer = action[1]
    
        error = desired_velocity - velocity
        
        proportional = self.kp * error 
        self.e = self.e + error * self.dt
        integral = self.ki * self.e 
        derivative = self.kd * ((error - self.e_prev) / self.dt)
        
        u = proportional + integral + derivative
        
        if u>= 0:
            throttle = max(min(u, 1), 0)
            brake = 0
        elif u<0:
            throttle = 0
            brake = -max(min(u, 0), -1)
        
        self.e_prev = self.e
        
        # safety brake.
        if desired_velocity<0.01:
            throttle=0
            brake=1
            
        return [throttle, steer, brake]
    
    def reset(self):
        self.e = 0
        self.e_prev = 0
        
