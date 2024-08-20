        
import numpy as np
from m3_model import CalcRelativeSaturationTime as CalcSaturationTimeExact
from m3_model import CalcFoldChange, CalcFoldChangeWholePopulation

def CalcRelativeYield(Ys, R0, N0):
    """Converts the biomass yield Y into dimensionless yield nu."""
    nus = 1 + (R0*Ys/N0)
    return nus
def CalcSaturationFrequencies(xs,fcs):
    """Returns the vector of lineage frequencies at saturation time."""
    nominator   = np.multiply(xs,fcs)
    denominator = nominator.sum()
    return nominator/denominator
def CalcTotalSelectionCoefficientLogit(xs,xs_final):
    "Returns total selection coefficient of each lineag wrt. total population"
    assert xs.shape == xs_final.shape, 'Both vectors need to have the same number of entries.'
    si = np.zeros_like(xs)
    
    ## first, manually pick strains where initial and final frequency are identical
    is_identical = xs == xs_final
    si[is_identical] = 0.
    ## then pick strains, where only the initial frequencies is equal to 1
    is_initial_one = (xs == 1) & (~is_identical)
    si[is_initial_one] = np.nan
    ## then pick strains, where only the final frequency is equal to 1
    is_final_one = (xs_final == 1) & (~is_identical)
    si[is_final_one] = np.nan
    ## for the rest, the computation is well-defined
    ### todo: also catch cases where one of the frequencies is zero
    is_well = (~is_initial_one) & (~is_final_one) & (~is_identical)
    si[is_well] = np.log(np.divide(xs_final[is_well],1-xs_final[is_well])) \
                - np.log(np.divide(xs[is_well],1-xs[is_well]))

    return si


def CalcTotalSelectionCoefficientLog(xs,xs_final):
    "Returns total selection coefficient of each lineag wrt. total population"
    assert xs.shape == xs_final.shape, 'Both vectors need to have the same number of entries.'
    si = np.zeros_like(xs)
    
    ## first, manually pick strains where initial and final frequency are identical
    is_identical = xs == xs_final 
    si[is_identical] = 0.
    
    ### for the rest, the computation is well-defined
    ### todo: also catch cases where one of the frequencies is zero
    si[~is_identical] = np.log(xs_final[~is_identical]) - np.log(xs[~is_identical])
    
    return si
def CalcPairwiseFrequency(xs):
    """Returns frequencies of all pairwise subpopulations using slow for loops."""
    xsum   = np.zeros((len(xs), len(xs)))
    xpairs = np.zeros((len(xs), len(xs)))
    for i in range(len(xs)):
        for j in range(len(xs)):
            xsum[i,j] = xs[i] + xs[j]
            xpairs[i,j] = xs[i]/xsum[i,j]
            
    ## set diagonal entries to 1
    np.fill_diagonal(xpairs,1)
            
    return xpairs

### use a vectorized version to speed up the process
def CalcPairwiseFrequencyFast(xs):
    """Returns frequencies of all pairwise subpopulations using fast matrix computation."""
    n = len(xs)
    xsum = np.multiply(xs, np.ones((n,n))).T + np.multiply(xs,np.ones((n,n)))
    xpairs = np.multiply(np.divide(1,xsum),xs).T
    ## fix diagonal entries
    np.fill_diagonal(xpairs,1)
    return xpairs
# calculate frequencies in subpopulation with a set of reference strains

def CalcReferenceFrequency(xs, ref_strains = [0] ):
    """Returns frequency wrt. to a set of reference strains. Use neutral strains as reference."""
    xgroup = xs[ref_strains].sum()
    xref   = np.divide(xs,xs+xgroup)
    
    # fix entries for the strains in the reference group themselves
    xref[ref_strains] = np.divide(xs[ref_strains],xgroup)
    return xref
    

def run_bulk_experiment(gs,ls,nus, xs):
    """Returns initial and final frequencies of all lineages in bulk competition growth cycle."""
   
    assert len(gs) == len(ls), "All trait vectors must have equal number of entries."
    assert len(gs) == len(nus), "All trait vectors must have equal number of entries."
    assert len(xs) == len(gs), "The frequency vector and trait vector must have same number of entries."

    # test the initial frequencies
    np.testing.assert_almost_equal(xs.sum(),1, decimal = 10, err_msg='The initial frequency of all strains must sum to one.')
    
    ## calculate
    tsat = CalcSaturationTimeExact(xs=xs,gs=gs, ls = ls, nus = nus)
    fcs = CalcFoldChange(t=tsat, g =gs, l = ls)
    #print(fcs)
    xs_final = CalcSaturationFrequencies(xs, fcs)
    
    # test the final frequencies
    np.testing.assert_almost_equal(xs_final.sum(),1, decimal = 10, err_msg='The final frequency of all strains must sum to one.')
    
    return xs, xs_final, tsat

def run_pairwise_experiment(gs,ls,nus, g1, l1, nu1, x0):
    """Returns initial and final frequencies of all strains in pairwise competition with wild-type."""
   
    assert len(gs) == len(ls), "All trait vectors must have equal number of entries."
    assert len(gs) == len(nus), "All trait vectors must have equal number of entries."
    assert x0 >0, "The initial frequency of the invading strain must be positive."


    ## calculate final frequencies
    x1, x2 = 1-x0,x0                    # rewrite initial frequency of the two strains
    xs_final = np.zeros_like(gs)        # final frequency
    tsats = np.zeros_like(gs)           # saturation time
    fcs_both = np.zeros_like(gs)       # foldchange of both species combined into one biomass
    fcs_wt = np.zeros_like(gs)          # foldchange of wildtype biomass
    fcs_mut = np.zeros_like(gs)          # foldchange of mut biomass

    for i in range(len(gs)):
        g2, l2, nu2 = gs[i], ls[i], nus[i] # get traits of the invader
        
        # compute joint biomass fold-change
        tsats[i] = CalcSaturationTimeExact(xs=[x1,x2], gs = [g1,g2], ls= [l1,l2], nus = [nu1,nu2] )
        fcs_both[i] = CalcFoldChangeWholePopulation(t = tsats[i],xs=[x1,x2], gs = [g1,g2], ls= [l1,l2])
        
        # compute individual biomass fold-change
        if x1 > 0:
            fcs_wt[i] = CalcFoldChange(t = tsats[i], g = g1, l = l1)
        else:
            fcs_wt[i] = 1 # defined to be 1, so frequency calculation works below
    
        if x2 > 0: 
            fcs_mut[i] = CalcFoldChange(t = tsats[i], g = g2, l = l2)
        else:
            fcs_mut[i] = 1

        # compute final frequency
        fcs1, fcs2 =  fcs_wt[i], fcs_mut[i] 
        xs_final[i] = x2*fcs2/(x1 *fcs1 + x2*fcs2)
        
        
    ### define initial frequencies of mutants as a vector
    xs = np.ones_like(xs_final)*x2
    

    return xs, xs_final, tsats, fcs_both, fcs_wt, fcs_mut

### conver per Cycle to Per_Cenerat

def toPerGeneration(s_percycle, fcs_wt):
    """Converts selection coefficient per-cycle to per-generation of wild-type."""
    W = np.divide(s_percycle,np.log(fcs_wt))
    return W

def CalcLenskiW(fcs_mut,fcs_wt):
    """Returns fitness statistic W as defined by Lensk et al. Am Nat 1991"""
    nominator = np.log(fcs_mut) 
    denominator= np.log(fcs_wt)
    W = np.nan*np.ones_like(fcs_mut) # set an original value, if division below fails
    W = np.divide(nominator,denominator, where = denominator !=0)
    return W


def CalcAbundanceTimeseries(t, g, l, tsat, N0):
    """Returns the abundance timeseries for a given strain under M3 growth."""

    ## calculate fold-change at saturation time
    fcsat = CalcFoldChange(t=tsat,g=g,l=l)
    
    ## add the lag time to the time vector
    tinput = np.hstack([t,[l]])
    tinput = np.sort(tinput,axis = 0)
    ## calculate abundance timeseries
    fcs  = np.ones_like(tinput)
    fcs  = np.where(tinput<tsat,CalcFoldChange(t=tinput,g=g,l=l),fcsat)
    y    = N0*fcs
    
    return tinput, y
                      
def CalcAreaUnderTheCurve(t,y,t_trim):
    """Calculates the AUC up to the timepoint t_trim.
    
    In particular, this computes the area between y=0 and the curve value y. T
    This is the smae  definition of the Area Under the Curve (AUC) as in the paper by
    
    Sprouffske & Wagner 2016 BMC Bioinformatics https://doi.org/10.1186/s12859-016-1016-7
    
    and their R-package 'Growthcurver'
    
    https://github.com/sprouffske/growthcurver/blob/25ea4b301311e95cbc8d18f5c30fb4c750782869/R/utils.R#L101
    
    """
    is_within = t <= t_trim
    return np.trapz(y[is_within], t[is_within])
    

#####

if __name__ == '__main__': 
    pass
    
