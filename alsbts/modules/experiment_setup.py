
from dataclasses import dataclass, field
import os

import numpy as np
from alts.core.configuration import Configurable
from alvsts.modules.consumer_behavior import ConsumerBehavior
from alvsts.modules.matlab_engin import MatLabEngin
from alvsts.modules.rvs_estimator import RVSEstimator
from alvsts.modules.change_detector import ChangeDetector

from typing_extensions import Self

class Simulation(Configurable):
    @property
    def is_running(self) -> bool:
        raise NotImplementedError()

    def init_simulation(self):
        raise NotImplementedError()

    def stop_simulation(self):
        raise NotImplementedError()


@dataclass
class ExperimentSetup(Simulation):
    eng: MatLabEngin = None
    consumer_behavior: ConsumerBehavior = None
    rvs_estimator: RVSEstimator = None
    change_detector: ChangeDetector = None
    path: str = "./"
    model_name: str="STmodel"
    sim_stop_time: float = 600

    def __call__(self, **kwargs) -> Self:
         obj: ExperimentSetup = super().__call__(**kwargs)
         obj.consumer_behavior = obj.consumer_behavior(stop_time = obj.sim_stop_time)
         obj.rvs_estimator = obj.rvs_estimator()
         obj.change_detector = obj.change_detector()
         obj.init_simulation()
         return obj

    def threshold_reached(self, log_likelihood: float):
        return log_likelihood < -25

    def observable_grid_quantities(self):
        return np.asarray((
            np.asarray(self.eng.workspace["timeOutput"])[-1][0],
            np.asarray(self.eng.workspace["voltageOutput"])[-1][0],
            np.asarray(self.eng.workspace["knewVOutput"])[-1][0],
            np.asarray(self.eng.workspace["activePowerOutput"])[-1][0],
            np.asarray(self.eng.workspace["reactivePowerOutput"])[-1][0],
        ))
    
    def trigger_disturbance(self):
        if self.is_running:
            self.eng.set_param(
                self.model_name
                + "/PWM outputs/voltage control/References/Manual Switch",
                "sw",
                "0",
                nargout=0,
            )
    
    def reset_disturbance_trigger(self):
        if self.is_running:
            self.eng.set_param(
                self.model_name
                + "/PWM outputs/voltage control/References/Manual Switch",
                "sw",
                "1",
                nargout=0,
            )

    def continue_sim(self):
            self.eng.set_param(
                self.model_name, "SimulationCommand", "continue", nargout=0
            )

    def trigger_vs_measurement(self):
        self.trigger_disturbance()
        self.continue_sim()
        self.reset_disturbance_trigger()

    def last_vs_measurement(self):
        measurement_in_progress = np.asarray(self.eng.workspace["TriggerSignalOutput"])[-1][0]
        measured_vs = np.asarray(self.eng.workspace["KpOutput"])[-1][0]
        return np.asarray((measurement_in_progress, measured_vs))

    def gt_vs(self):
        vs = np.asarray(self.eng.workspace["VS_GT"])[-1][0]
        return vs
    
    def estimate_rvs(self, vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput):
        rvs = self.rvs_estimator.estimate(vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput)
        return rvs

    def rvs_change(self, vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs):
        change_point = self.change_detector.detect(vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs)
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
        
        kp_str = "".join(np.array2string(ts_kp, precision=4, suppress_small=True, separator=',',threshold=np.inf, max_line_width=np.inf))
        change_time_str = "".join(np.array2string(ts_change_time, precision=4, suppress_small=True, separator=',',threshold=np.inf, max_line_width=np.inf))
        kpwork_test_data = ("Kpwork = timeseries2timetable(timeseries(" +kp_str + "," + change_time_str + "));").replace("\n", "")
        with open(os.path.join(self.path,"Kpwork_f.m"), "w") as f:
            f.write(kpwork_test_data)
        self.eng.run(os.path.join(self.path,"Kpwork_f.m"), nargout=0)
    
    def init_simulation(self):
        model_path = os.path.join(self.path,f'{self.model_name}.slx')
        print("loading from:",os.path.abspath(model_path))
        # Load model settings
        self.eng.run(os.path.join(self.path,"init_STsetup.m"), nargout=0)

        self.init_consumer_behavior()

        self.eng.workspace['stop_time'] = float(self.sim_stop_time)

        # Load the model
        self.eng.eval(f"model = '{model_path}';", nargout=0)
        self.eng.eval("load_system(model)", nargout=0)

        # Start, simulation
        self.eng.set_param(
            self.model_name,
            "SimulationCommand",
            "start",
            nargout=0,
        )

        # Simulation immediately pauses its self after first time step
        # continue sim for a few more steps so that transient process is over and sim is stable
        # skip n sim steps
        for i in range(25):
            self.continue_sim()

    

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
    

    def estimate_vs(self, currentTime):
            self.calculationTimes.append(currentTime)
            currentRelativeLoad = self.getCurrentRelativeLoad(currentTime)
            self.relativeLoads.append(currentRelativeLoad)

            self.kf.predict(currentRelativeVS=currentRelativeLoad)
            return self.kf.x
