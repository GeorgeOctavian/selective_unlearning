import pandas as pd
from pytorch_lightning import Callback

class MetricTracker(Callback):

    def __init__(self, run_name):
        self.df = None
        self.run_name = run_name

    def on_validation_end(self, trainer, module):
        print(trainer.logged_metrics)
        elogs = trainer.logged_metrics # access it here
        elogs = {k: [v.item()] for k, v in elogs.items()}
        new_df = pd.DataFrame(elogs)
        new_df.to_csv(f'experiments/csv_outputs/{self.run_name}.csv')


    # on_training_end?