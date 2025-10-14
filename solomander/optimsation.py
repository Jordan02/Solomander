import optuna
import optuna.visualization.matplotlib as vism
import pandas as pd
import io

import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk as cyberpunk

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

        self.pnl_curve = []
        self.display_values = []

    def execute(self):

        self.study = optuna.create_study(direction=self.direction, sampler=optuna.samplers.TPESampler())
        
        if self.guess is not None:
            for g in self.guess:
                self.study.enqueue_trial(g)

        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        return self.study.best_params
        
    def show(self, pnl_curves=None , info = None, discord=False, color = "#10f7ff"):

        BEST_COLOR = color
        BEST_ARROW_COLOR = color
        GUESS_COLOR = "#88888840"
        ARROW_COLOR = "#ffffff"

        TOP_OFFSET = 0.95  
        BOTTOM_OFFSET = 0.05  
        SPACING = 0.05

        available = TOP_OFFSET - BOTTOM_OFFSET
        n = len(self.guess) + 1
        if self.guess == []: n += 1

        vh = (available - (n-1)*SPACING)/n
        v_pos = lambda i: (TOP_OFFSET-vh*i - vh/2 - SPACING*i)

        
        best_idx = self.study.best_trial.number

        # ===== plots =====
        plt.style.use("cyberpunk")
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
            ax_pnl.plot(np.arange(len(pnl_curves[best_idx])), pnl_curves[best_idx], label="Best Trial", color = BEST_COLOR, lw=2)
            textstr = "\n".join([f"{k}: {v:<.2f}" for k, v in info[best_idx].items()])

            x_pos = len(pnl_curves[best_idx]) - 1
            y_pos = pnl_curves[best_idx][-1]

            ax_pnl.annotate(textstr, 
                            xy=(x_pos, y_pos),
                            xycoords="data",   
                            xytext=(1.02, v_pos(0)), #top right outside axes
                            textcoords="axes fraction",
                            ha="left",
                            va="center", 
                            transform=ax_pnl.transAxes, #relative to axes
                            fontsize=7,
                            verticalalignment='top',
                            color = BEST_COLOR,
                            bbox=dict(boxstyle="round,pad=0.3", edgecolor=BEST_ARROW_COLOR, facecolor = "none", alpha=0.9, lw=1),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=BEST_ARROW_COLOR))
            
            cyberpunk.make_lines_glow(ax_pnl)

            # ===== PLOT INITIAL GUESSES =====
            if self.guess == []:
                self.guess.append(self.study.trials[0].params)
                pass
            
            # create annotations for each guess, storing x and y positions
            annotations = [
                            {"text": "\n".join([f"{k}: {v:<.2f}" for k, v in info[i].items()]),
                            "x": len(curve) - 1,
                            "y": curve[-1]
                            } 
                            for i, curve in enumerate(pnl_curves[:len(self.guess)])    
                        ]
            
            # sort annotations by y value (highest first)
            annotations.sort(key=lambda ann: ann["y"], reverse=True)  # sort by y value to minimize overlap

            for i, ann in enumerate(annotations):
                ax_pnl.plot(np.arange(len(pnl_curves[i])), pnl_curves[i], label=f"Guess {i+1}", color=GUESS_COLOR, lw=1, alpha=0.7)
                ax_pnl.annotate(ann["text"],
                                xy=(ann["x"], ann["y"]),
                                xycoords="data",
                                xytext=(1.02, v_pos(i+1)), #top right outside axes
                                textcoords="axes fraction",
                                ha="left",
                                va="center", 
                                fontsize=7,
                                bbox=dict(boxstyle="round,pad=0.3", edgecolor=GUESS_COLOR, facecolor="none", alpha=0.6),
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=ARROW_COLOR))
        
        # ===== Plotting test optimsation serach =====
        x = [t.number for t in self.study.trials]
        y = [t.value for t in self.study.trials]

        if self.study.direction.name == "MAXIMIZE":
            best_so_far = np.maximum.accumulate(y)
        else:
            best_so_far = np.minimum.accumulate(y)

        ax_study.plot(x, best_so_far, color=BEST_COLOR, lw=1, label="Best Value")
        ax_study.scatter(x, y, s=10, alpha=0.7, label="Objective Value",color = GUESS_COLOR)
        ax_study.legend()
        ax_pnl.legend()
        ax_pnl.legend().remove()

        plt.tight_layout(pad=0.5)
        cyberpunk.make_lines_glow(ax_study)
        ax_pnl.grid(True, alpha=0.2, color="#ffffff")
        ax_study.grid(True, alpha=0.2, color="#ffffff")

        # remove background
        if discord:
            fig.patch.set_alpha(0.0)       
            ax_pnl.set_facecolor("none")
            ax_study.set_facecolor("none")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", transparent=True, dpi=200)
            buf.seek(0)
            plt.close(fig)

            return buf  # return the BytesIO buffer for Discord sending
        else:
        
            plt.show()
            return fig