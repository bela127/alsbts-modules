
from dataclasses import dataclass
import os

import numpy as np
from alvsts.modules.consumer_behavior import ConsumerBehavior
from alvsts.modules.matlab_engin import MatLabEngin

class Simulation():
    def is_running(self) -> bool:
        raise NotImplementedError()

@dataclass
class ExperimentSetup(Simulation):
    eng: MatLabEngin = None
    consumer_behavior: ConsumerBehavior = None
    path: str = "./"
    model_name: str="STmodel"


    def __post_init__(self) -> None:
         self.init_simulation()

    def threshold_reached(self, log_likelihood: float):
        return log_likelihood < -25

    def observable_grid_quantities(self):
        return (
            np.asarray(self.eng.workspace["voltageOutput"]),
            np.asarray(self.eng.workspace["knewVOutput"]),
            np.asarray(self.eng.workspace["activePowerOutput"]),
            np.asarray(self.eng.workspace["reactivePowerOutput"]),
            np.asarray(self.eng.workspace["timeOutput"]),
        )
    
    def make_disturbance(self):
        if self.is_running():
            self.eng.set_param(
                self.model_name
                + "/PWM outputs/voltage control/References/Manual Switch",
                "sw",
                "0",
                nargout=0,
            )
            self.eng.set_param(
                self.model_name, "SimulationCommand", "continue", nargout=0
            )
            self.eng.set_param(
                self.model_name, "SimulationCommand", "pause", nargout=0
            )

    def measure_vs(self):
        measure_vs = True
        while measure_vs and self.is_running:
                    self.make_disturbance()

                    #get disturbance trigger (if its finished its 0)
                    currentTriggerSignal = self.eng.workspace["TriggerSignalOutput"][-1]

                    # if voltage sensitivity calculation via disturbance is done:
                    if currentTriggerSignal[0] == 0.0:
                        currentTime = self.eng.workspace["timeOutput"][-1]

                        measured_vs = self.eng.workspace["KpOutput"][-1]
                        measure_vs = False
        return currentTime, measured_vs #type: ignore
    
    def estimate_rvs(self, time):
        rvs = 1
        return rvs

    def rvs_change(self, time):
        change_point = 0
        return change_point
        
    @property
    def is_running(self):
        return self.eng.get_param(self.model_name, "SimulationStatus") != ("stopped" or "terminating")

    def init_consumer_behavior(self):
        change_time, kp = self.consumer_behavior.behavior()

        # Make time series of constant kps from kp values
        ts_change_time = np.zeros(len(change_time) * 2)
        ts_kp = np.zeros(len(kp) * 2)
        for i in range(0, len(change_time)):
            ts_change_time[2 * i] = change_time[i]
            ts_change_time[2 * i + 1] = change_time[i]

            ts_kp[2 * i + 1] = kp[i]
            if i == 0:
                ts_kp[2 * i] = kp[i]
            else:
                ts_kp[2 * i] = kp[i - 1]
        
        kp_str = "".join(str(ts_kp))
        change_time_str = "".join(str(ts_change_time))
        kpwork_test_data = ("Kpwork = timeseries(" +kp_str + "," + change_time_str + ");").replace("\n", "")
        with open(os.path.join(self.path,"Kpwork_f.m"), "w") as f:
            f.write(kpwork_test_data)
        self.eng.run(os.path.join(self.path,"Kpwork_f.m"), nargout=0)
    
    def init_simulation(self):
        # Load model settings
        self.eng.run(os.path.join(self.path,"init_STsetup.m"), nargout=0)

        self.init_consumer_behavior()

        # Load the model
        self.eng.eval(f"model = '{os.path.join(self.path,f'{self.model_name}.slx')}'", nargout=0)
        self.eng.eval("load_system(model)", nargout=0)

        # Start, then immediately pause the simulation after first time step
        self.eng.set_param(
            self.model_name,
            "SimulationCommand",
            "start",
            "SimulationCommand",
            "pause",
            nargout=0,
        )

    def stop_simulation(self):
        self.eng.set_param(self.model_name, "SimulationCommand", "stop", nargout=0)

    def update_models(self, currentTime, measured_vs):
                currentRelativeLoad = self.estimate_rvs(currentTime)
                self.xArr.append(currentRelativeLoad)

                self.yArr.append(measured_vs)
                print("Current Calculated Load: ", measured_vs)

                self.model.fit(np.asarray(self.xArr), np.asarray(self.yArr))
                self.disturbanceTimes.append(currentTime)

                # update after measurement
                self.kf.update(measured_vs)
                print("Measurement Update on KF!")
                print("new x:  ", self.kf.x)

    def simulation_loop(self):
        while self.is_running():
            self.simulation_step()

    def simulation_step(self):
        # Run the Simulation, then pause again
        self.eng.set_param(
            self.model_name,
            "SimulationCommand",
            "continue",
            nargout=0,
        )
        # reset disturbance switch
        self.eng.set_param(
            self.model_name
            + "/PWM outputs/voltage control/References/Manual Switch",
            "sw",
            "1",
            nargout=0,
        )

        self.eng.set_param(
            self.model_name,
            "SimulationCommand",
            "pause",
            nargout=0,
        )

        # currentTime = self.eng.workspace["timeOutput"][-1]
        # if np.float32(currentTime) > self.updateTime:
        #     print(currentTime)
        #     vs_estimate = self.estimate_vs(currentTime)

        #     self.ukfStates.append(vs_estimate)
        #     self.updateTime += 5.0

        # if self.threshold_reached(self.kf.log_likelihood):
        #     print("log likelihood:  ", self.kf.log_likelihood)
            
        #     currentTime, measured_vs = self.measure_vs()

        #     self.update_models(currentTime, measured_vs)
    

    def estimate_vs(self, currentTime):
            self.calculationTimes.append(currentTime)
            currentRelativeLoad = self.getCurrentRelativeLoad(currentTime)
            self.relativeLoads.append(currentRelativeLoad)

            self.kf.predict(currentRelativeVS=currentRelativeLoad)
            return self.kf.x
