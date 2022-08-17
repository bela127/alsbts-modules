from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
from typing_extensions import Self
from alts.core.evaluator import LogingEvaluator, Evaluate

import numpy as np
from matplotlib import pyplot as plot # type: ignore
import os


if TYPE_CHECKING:
    from typing import List, Tuple
    from alts.core.experiment import Experiment
    from nptyping import  NDArray, Number, Shape


@dataclass
class PlotVSEvaluator(LogingEvaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "VS"

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.oracle.data_source.exp_setup.observable_grid_quantities = Evaluate(self.experiment.oracle.data_source.exp_setup.observable_grid_quantities)
        self.experiment.oracle.data_source.exp_setup.observable_grid_quantities.post(self.save_gq)

        self.experiment.oracle.data_source.exp_setup.gt_vs = Evaluate(self.experiment.oracle.data_source.exp_setup.gt_vs)
        self.experiment.oracle.data_source.exp_setup.gt_vs.post(self.save_gt_vs)

        self.experiment.oracle.data_source.exp_setup.estimate_rvs = Evaluate(self.experiment.oracle.data_source.exp_setup.estimate_rvs)
        self.experiment.oracle.data_source.exp_setup.estimate_rvs.post(self.save_rvs)

        self.experiment.oracle.data_source.query = Evaluate(self.experiment.oracle.data_source.query)
        self.experiment.oracle.data_source.query.pre(self.save_query)

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.plot_data)

        self.gq = None
        self.vs = []
        self.rvs = []
        self.query = None

    def save_gq(self, gq):
        if self.gq is None:
            self.gq = np.asarray([gq])
        else:
            self.gq = np.concatenate((self.gq, np.asarray([gq])))

    def save_gt_vs(self, vs):
        self.vs.append(vs)

    def save_rvs(self, rvs):
        self.rvs.append(rvs)
    
    def save_query(self, query):
        if self.query is None:
            self.query = query
        else:
            self.query = np.concatenate((self.query, query))

    def plot_data(self, exp_nr):
        time = self.gq[:,0]
        gt_vs = self.vs
        vs_estimation = self.query[:, 0]
        rvs = self.rvs
        measure = self.query[:,1]



        fig = plot.figure(self.fig_name)
        plot.plot(time,gt_vs)
        plot.plot(time,vs_estimation)
        plot.plot(time,rvs)
        plot.scatter(time,measure)
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.path}/{self.fig_name}_{exp_nr:05d}.png')
            plot.clf()


@dataclass
class LogAllEvaluator(LogingEvaluator):
    folder: str = "log"
    file_name:str = "all_data_source_data"

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.oracle.data_source.exp_setup.observable_grid_quantities = Evaluate(self.experiment.oracle.data_source.exp_setup.observable_grid_quantities)
        self.experiment.oracle.data_source.exp_setup.observable_grid_quantities.post(self.save_gq)

        self.experiment.oracle.data_source.exp_setup.gt_vs = Evaluate(self.experiment.oracle.data_source.exp_setup.gt_vs)
        self.experiment.oracle.data_source.exp_setup.gt_vs.post(self.save_gt_vs)

        self.experiment.oracle.data_source.exp_setup.estimate_rvs = Evaluate(self.experiment.oracle.data_source.exp_setup.estimate_rvs)
        self.experiment.oracle.data_source.exp_setup.estimate_rvs.post(self.save_rvs)

        self.experiment.oracle.data_source.exp_setup.rvs_change = Evaluate(self.experiment.oracle.data_source.exp_setup.rvs_change)
        self.experiment.oracle.data_source.exp_setup.rvs_change.post(self.save_change)

        self.experiment.oracle.data_source.exp_setup.last_vs_measurement = Evaluate(self.experiment.oracle.data_source.exp_setup.last_vs_measurement)
        self.experiment.oracle.data_source.exp_setup.last_vs_measurement.post(self.save_measured_vs)

        self.experiment.oracle.data_source.query = Evaluate(self.experiment.oracle.data_source.query)
        self.experiment.oracle.data_source.query.pre(self.save_query)

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_data)

        self.gq = None
        self.vs = []
        self.rvs = []
        self.change = []
        self.query = None
        self.measured_vs = None

    def save_gq(self, gq):
        if self.gq is None:
            self.gq = gq[None,:]
        else:
            self.gq = np.concatenate((self.gq, gq[None,:]))

    def save_gt_vs(self, vs):
        self.vs.append(vs)

    def save_rvs(self, rvs):
        self.rvs.append(rvs)

    def save_change(self, change):
        self.change.append(change)

    
    def save_query(self, query):
        if self.query is None:
            self.query = query
        else:
            self.query = np.concatenate((self.query, query))

    def save_measured_vs(self, measured_vs):
        if self.measured_vs is None:
            self.measured_vs = measured_vs[None, :]
        else:
            self.measured_vs = np.concatenate((self.measured_vs, measured_vs[None, :]))

    def log_data(self, exp_nr):

        data = np.concatenate((self.gq, np.asarray(self.vs)[:,None], self.query, np.asarray(self.rvs)[:,None], np.asarray(self.change)[:,None], self.measured_vs), axis=1)

        np.save(f'{self.path}/{self.file_name}.npy', data)
