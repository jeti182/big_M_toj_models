import pymc3 as  pm
import pandas as pd
from matplotlib.pylab import plt

from models import logistic_regression_model, tvatoj_model, aqgp_model
from score_and_plot import plot_ppc_and_score

df = pd.read_csv('dataset.csv')  
all_participants = sorted(set(df['PARTICIPANT_NUMBER']))

# For all participants (in subsets of 2) ...
for ps in [(0,2),(2,4),(4,6),(6,8)]:

    # Select subset of the data
    participants = all_participants[ps[0]:ps[1]]

    # Create empty figure
    f, axs = plt.subplots(3, 2, sharex=True, figsize=(6,8))

    # Sample from each model and create plots
    for i,p in enumerate(participants):
        
        # Exclude observations where fixation was lost
        data = df[(df['PARTICIPANT_NUMBER'] == p) & (df['EYE_ERROR'] == 0)]
        
        # Run logistic regression model
        with logistic_regression_model(data) as _lr_model:
            lr_trace = pm.sample(8000, tune=2000, init='adapt_diag', chains=4)
            plot_ppc_and_score(lr_trace, data, paras=['PSS', 'DL'],
                               title='P'+str(p)+': Logistic Regression', ax=axs[0,i])
        del _lr_model, lr_trace # Just to free up memory. 
        # You might consider saving these objects to disk for later use.

        # Run TVATOJ model
        with tvatoj_model(data) as _tvatoj_model:
            tvatoj_trace = pm.sample(8000, tune=2000, init='adapt_diag', chains=4)
            plot_ppc_and_score(tvatoj_trace, data, paras=['C', 'w_p'],
                                title='P'+str(p)+': TVATOJ', ax=axs[1,i])
        del _tvatoj_model, tvatoj_trace 

        # Run AQGP model
        with aqgp_model(data) as _aqgp_model:
            aqgp_trace = pm.sample(8000, tune=4000, init='adapt_diag', chains=4,
                                   target_accept=0.95) 
            plot_ppc_and_score(aqgp_trace, data, 
                                paras=['λ_p', 'λ_r', 'Δ', 'τ', 'ξ','ε_p', 'ε_r'],
                                title='P'+str(p)+': AQGP', ax=axs[2,i])
        del _aqgp_model, aqgp_trace

        # Save plot to file
        plt.tight_layout()
        plt.savefig('participants-%d-to-%d.svg'%(ps[0],ps[1]))