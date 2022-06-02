import pymc3 as  pm
from theano import tensor as tt

########## Psychometric functions ##########

def difcdf(x, shift, rp, rr):
    """ Bilateral exponential CDF arrival time distribution with two
        rates rp and rr and a shift parameter as defined by
        Alcalá-Quintana & García-Pérez (2013) 
        [Behav Res Methods doi.org/10.3758/s13428-013-0325-2]
    """
    y = x - shift
    left = rp * tt.exp(rr * y) / (rp + rr)
    right = 1 - (rr * tt.exp(-rp * y) / (rp + rr))
    return (y <= 0) * left + (y > 0) * right

def aqgp(soa, λ_p, λ_r, Δ, τ, ξ, ε_p, ε_r):
    """ Psychometric functions from Alcalá-Quintana & García-Pérez (2013) 
        See source for parameter meanings
    """
    pPF = difcdf(-Δ, soa+τ, λ_p, λ_r)
    pRF = 1 - difcdf(Δ, soa+τ, λ_p, λ_r)
    pS = 1 -pPF - pRF
    return (1 - ε_p) * pPF + (1 - ξ) * pS + ε_r * pRF

def tvatoj(soa, C, wp):
    """ TVA-based psychometric function parametrized via difcdf (see above).
        For parameter meaning see Tünnermann, Petersen & Scharlau (2015)
        [J Vis //doi.org/10.1167/15.3.1 ] or Krüger et al. (2021). 
    """
    rp = C * wp
    rr = C * (1 - wp)
    return 1-difcdf(soa, 0, rr, rp) 
    
########## Handle data ##########

def provide_data(data):
    """ Extract rows from long dataframe. Modify to use with other formats."""
    soas  = data['SOA_IN_MS'].values
    pf  = data['PROBE_FIRST_RESPONSE'].values
    condition  = data['PROBE_SALIENT'].values 
    return (soas, pf, condition)

########## Graphical models (PyMC3 implementations) ########## 

def logistic_regression_model(data):
    """ Uses PyMC3's default logistic regression with its default priors """

    soas, pf, condition = provide_data(data)

    with pm.Model() as lr_model:
        # Model is a one-liner!
        formula = \
        'PROBE_FIRST_RESPONSE ~ SOA_IN_MS + PROBE_SALIENT + SOA_IN_MS * PROBE_SALIENT'
        pm.glm.GLM.from_formula(formula, data, family=pm.glm.families.Binomial())

        # Deterministic transforms for compatibly with the visualization & PSS + DL
        a = pm.Deterministic('a', tt.stack((lr_model['Intercept'],
                              lr_model['Intercept']+lr_model['PROBE_SALIENT'])))
        b = pm.Deterministic('b', tt.stack((lr_model['SOA_IN_MS'], 
                              lr_model['SOA_IN_MS']+lr_model['SOA_IN_MS:PROBE_SALIENT'])))
        PSS = pm.Deterministic('PSS', -a/b)
        DL = pm.Deterministic('DL',  (tt.log(0.75/0.25)/b))

    return lr_model

def tvatoj_model(data):
    """ The TVA-TOJ model with default priors motivated here"""

    soas, pf, condition = provide_data(data)

    with pm.Model() as tvatoj_model:
        C = pm.Normal('C', 0.08, 0.05) 
        w_p = pm.Normal('w_p', 0.5, 0.2, shape=2) 
        θ = pm.Deterministic('θ', tvatoj(data['SOA_IN_MS'].values, C , w_p[condition]))
        y = pm.Bernoulli('y', p=θ, observed=data['PROBE_FIRST_RESPONSE'])
    
    return tvatoj_model

def aqgp_model(data):
    """ Alcalá-Quintana & García-Pérez's (2013) full 7-parameter version """

    soas, pf, condition = provide_data(data)

    with pm.Model() as aqgp_model:
        λ_p = pm.Normal('λ_p', 0.04, 0.02, shape=2)
        λ_r = pm.Normal('λ_r', 0.04, 0.02, shape=2)
        Δ = pm.Uniform('Δ', 0, 100, shape=2)
        τ = pm.Normal('τ', 0, 30, shape=2)
        ξ = pm.Normal('ξ', 0.5, 0.2, shape=2)
        ε_p = pm.HalfCauchy('ε_p', 0.05, shape=2)
        ε_r = pm.HalfCauchy('ε_r', 0.05, shape=2)
        θ = pm.Deterministic('θ', aqgp(soas, λ_p[condition], λ_r[condition], 
                                             Δ[condition], τ[condition], ξ[condition],
                                             ε_p[condition], ε_r[condition]))

        y = pm.Bernoulli('y', p=θ, observed=pf)
    return aqgp_model