import pymc3 as pm
import arviz as az 
import numpy as np
from matplotlib.pylab import plt, rc
rc('font', size=6); rc('lines', linewidth=1); rc('lines', markersize=2)

def plot_ppc_and_score(trace, data, ax=None, title='PPC', paras=None):

    # Sample PPC
    ppc_trace = pm.sample_posterior_predictive(trace=trace, var_names=['y'])

    # Calculate LOO score
    loo = pm.loo(trace).loo
    loo_text = "LOO = %.2f"%loo

    # Aggregate binary responses
    new_trace = []
    print(sorted(set((data.SOA_IN_FRAMES))))
    for soa in sorted(set((data.SOA_IN_FRAMES))):
        new_trace.append(ppc_trace['y'][:,(data.SOA_IN_FRAMES==soa) & 
                                        (data.PROBE_SALIENT==0)].mean(axis=1))
        new_trace.append(ppc_trace['y'][:,(data.SOA_IN_FRAMES==soa) & 
                                        (data.PROBE_SALIENT==1)].mean(axis=1))
    ppc_trace = {'y': np.array(new_trace).T}
        
    # Prepare axes if none provided
    if ax is None: f,ax= plt.subplots() 

    # Get SOAs and condition mask from data
    SOAs = sorted(set(data['SOA_IN_MS'])) 
    cond  = data.groupby(['SOA_IN_MS', 'PROBE_SALIENT'])['PROBE_SALIENT'].min().values 

    # Plot
    az.plot_hdi(y=ppc_trace['y'][:,cond==0],x=SOAs, color='k', ax=ax, 
                hdi_prob=0.95, fill_kwargs={'alpha' : 0.23})  
    az.plot_hdi(y=ppc_trace['y'][:,cond==1],x=SOAs, color='r', ax=ax, 
                hdi_prob=0.95, fill_kwargs={'alpha' : 0.23})  
    ax.plot(SOAs, np.mean(ppc_trace['y'][:,cond==0],axis=0), color='k')  
    ax.plot(SOAs, np.mean(ppc_trace['y'][:,cond==1],axis=0), color='r')  
    pfs = data.groupby(['SOA_IN_MS', 'PROBE_SALIENT']).mean()['PROBE_FIRST_RESPONSE']
    ax.plot(SOAs, pfs.values[::2], 'k.')   
    ax.plot(SOAs, pfs.values[1::2], 'r.')   
    ax.axvline(0.5, linestyle='dashed')
    ax.axhline(max(pfs)/2, linestyle='dashed')
    ax.text(-20,0, loo_text)
    if paras is not None:
        for i, varname in enumerate(paras):
            stats = pm.summary(trace, var_names=[varname], hdi_prob=.95)  
            for j, s in enumerate(stats['mean']):
                text = r'$' + varname + r'$: %.2f [%.2f, %.2f]'
                text = text%(s, stats['hdi_2.5%'][j], stats['hdi_97.5%'][j])
                posy, posy = .1 + .5 - (1 - j) * .5, 0.95 - (.05*i) - ((1-j)*.5)
                ax.text(posx, posy, text, transform = ax.transAxes, color=['k','r'][j])
    ax.set_title(title)