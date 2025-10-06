import optuna
import optuna.visualization.matplotlib as vism
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


try:
    from .logger import log, stamp, pront
    from .baseStrategy import Strategy
except ImportError:
    from logger import log, stamp, pront
    from solomander.baseStrategy import Strategy


class Optimise:

    study: optuna.study.Study

    def __init__(self, objective, n_trials=50, n_jobs=1, guess=[], direction="maximize", target="Objective Value", **kwargs):
        
        self.objective = objective
        self.kwargs = kwargs
        
        self.target = target
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.direction = direction
        self.guess = guess

    def execute(self):

        self.study = optuna.create_study(direction=self.direction, sampler=optuna.samplers.TPESampler())
        
        if self.guess is not None:
            for g in self.guess:
                self.study.enqueue_trial(g)

        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        return self.study.best_params
        
    def show(self, pnl_curves=None , info = None):

        best_idx = self.study.best_trial.number

        # ===== plots =====
        fig = plt.figure(figsize=(10,5))
        gs = fig.add_gridspec(2,2, width_ratios=[1,1])
        ax_pnl = fig.add_subplot(gs[:,1])
        ax_study = fig.add_subplot(gs[:,0])
        
        
        ax_study.set_xlabel("Trial Number")
        ax_study.set_ylabel(self.target)
        ax_pnl.set_xlabel("Trades")
        ax_pnl.set_ylabel("PnL")

        if pnl_curves and info is not None:
          
            # ===== PLOT BEST CURVE =====
            ax_pnl.plot(np.arange(len(pnl_curves[best_idx])), pnl_curves[best_idx], label="Best Trial")
            textstr = "\n".join([f"{k}: {v:<.2f}" for k, v in info[best_idx].items()])

            x_pos = len(pnl_curves[best_idx]) - 1
            y_pos = pnl_curves[best_idx][-1]

            ax_pnl.annotate(textstr, 
                            xy=(x_pos, y_pos),
                            xytext=(x_pos + 10, y_pos -40),
                            xycoords="data",   
                            textcoords="data",
                            ha="left",
                            va="bottom", 
                            transform=ax_pnl.transAxes, #relative to axes
                            fontsize=6,
                            verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
                           
            # ===== PLOT INITIAL GUESSES =====
            if self.guess == []:
                self.guess.append(self.study.trials[0].params)
                pass

            for i, g in enumerate(self.guess):
                ax_pnl.plot(np.arange(len(pnl_curves[i])), pnl_curves[i], label=f"Guess {i+1}", color="#88888840")
                
                textstr = "\n".join([f"{k}: {v:<.2f}" for k, v in info[i].items()])
                x_pos = len(pnl_curves[i]) - 1
                y_pos = pnl_curves[i][-1]

                ax_pnl.annotate( textstr, 
                                xy=(x_pos, y_pos),
                                xytext=(x_pos + 10, y_pos -40),
                                xycoords="data",   
                                textcoords="data",
                                ha="left",
                                va="top", 
                                fontsize=6,
                                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7),
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
        
        # ===== Plotting test optimsation serach =====
        x = [t.number for t in self.study.trials]
        y = [t.value for t in self.study.trials]

        if self.study.direction.name == "MAXIMIZE":
            best_so_far = np.maximum.accumulate(y)
        else:
            best_so_far = np.minimum.accumulate(y)

        ax_study.plot(x, best_so_far, color="orange", lw=1, label="Best Value")
        ax_study.scatter(x, y, s=10, alpha=0.7, label="Objective Value")
        ax_study.legend()
        ax_pnl.legend()
        #see = vism.plot_optimization_history(self.study)
        plt.show()