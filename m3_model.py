import numpy
from numpy import random
import scipy.special
import scipy.integrate
import matplotlib.colors
import warnings
warnings.filterwarnings("error")
#from itertools import product, combinations_with_replacement


################################################################################
################################################################################
############################# AUXILIARY FUNCTIONS ##############################
################################################################################
################################################################################


################################################################################
#    Produces a new color by add alpha to the input hex color.
################################################################################
def alpha_blending(hex_color, alpha):
    foreground_tuple  = matplotlib.colors.hex2color(hex_color)
    foreground_arr = numpy.array(foreground_tuple)
    final = tuple( (1. -  alpha) + foreground_arr*alpha )
    return(final)


################################################################################
#    Various colors
################################################################################
BLUE = "#0770b5"
RED = "#c52424"
GREEN = "#336600"
ORANGE = "#db8503"
VIOLET = "#783d72"

LIGHT_BLUE = alpha_blending(BLUE, 2/3)
LIGHT_RED = alpha_blending(RED, 2/3)
LIGHT_GREEN = alpha_blending(GREEN, 2/3)
LIGHT_ORANGE = alpha_blending(ORANGE, 2/3)
LIGHT_VIOLET = alpha_blending(VIOLET, 2/3)

VERY_LIGHT_BLUE = alpha_blending(BLUE, 1/3)
VERY_LIGHT_RED = alpha_blending(RED, 1/3)
VERY_LIGHT_GREEN = alpha_blending(GREEN, 1/3)
VERY_LIGHT_ORANGE = alpha_blending(ORANGE, 1/3)
VERY_LIGHT_VIOLET = alpha_blending(VIOLET, 1/3)


################################################################################
#    This function returns the sign of a float, or zero if the float is very
#    close to zero.
################################################################################
def Sign(x):
     if x > 1e-15:       return 1
     elif x < -1e-15:    return -1
     else:               return 0


################################################################################
#    Calculates the harmonic mean of a list of numbers.
################################################################################
def HarmonicMean(xs):
     return len(xs)/sum(1/x for x in xs)


################################################################################
#    Calculates the Heaviside theta (step) function.
################################################################################
def HeavisideTheta(x):
     return 1*(x > 0)


################################################################################
#    Calculates the total variation distance between two sets of normalized 
#    frequencies.
################################################################################
def CalcFreqDistance(xs1, xs2):
     return 0.5*sum(abs(xs1[i] - xs2[i]) for i in range(len(xs1)))
     #return numpy.sqrt(sum((xs1[i] - xs2[i])**2 for i in range(len(xs1))))


################################################################################
#    Generates two lists of traits from a bivariate Gaussian distribution.
################################################################################
def GenerateGaussianTraits(x_mean, x_sd, y_mean, y_sd, rho, num_types):
     cov = numpy.array([[x_sd**2, rho*x_sd*y_sd], [rho*x_sd*y_sd, y_sd**2]])
     xs, ys = numpy.random.multivariate_normal([x_mean, y_mean], cov, num_types).T
     return xs, ys


################################################################################
#    Generates a list of traits from a single gamma distribution.
################################################################################
def GenerateGammaTraits(mean, sd, num_types):
     return numpy.random.gamma((mean/sd)**2, sd**2/mean, num_types)


################################################################################
################################################################################
####################### FUNCTIONS FOR EXACT GROWTH MODEL #######################
################################################################################
################################################################################


################################################################################
#    Calculates the fold-change at time t for a strain with growth rate g and 
#    lag time l.
################################################################################
def CalcFoldChange(t, g, l):
     ## convert to arrays
     gs = numpy.array(g)
     ls = numpy.array(l)
     return numpy.exp(numpy.multiply(gs,t - ls)*HeavisideTheta(t - ls))


################################################################################
#    Calculates the fold-change at time t for a mixed population with growth 
#    rates gs and lag times ls.
################################################################################
def CalcFoldChangeWholePopulation(t, xs, gs, ls):
     fcs = CalcFoldChange(t=t,g=gs,l=ls)
     return sum(numpy.multiply(xs,fcs))/sum(xs)


################################################################################
#    Calculates the instantaneous population growth rate at time t.
################################################################################
def CalcPopGrowthRateExact(t, xs, gs, ls):
     try:
          numerator = sum(gs*HeavisideTheta(t - ls)*numpy.exp(gs*(t - ls)*HeavisideTheta(t - ls)))
          denominator = sum(numpy.exp(gs*(t - ls)*HeavisideTheta(t - ls)))
     except RuntimeWarning:
          print("Caught warning.  Skipping this realization")
          return None

     return numerator/denominator


################################################################################
#    Calculates the total remaining fraction of resources at time t given 
#    strains with initial fractions xs, growth rates gs, lag times ls, and 
#    relative yields nus.
################################################################################
def CalcTotalResources(t, xs, gs, ls, nus):
     return sum(xs[i]*CalcFoldChange(t, gs[i], ls[i])/nus[i] for i in range(len(xs)))


################################################################################
#    Calculates the relative saturation time of multiple strains with initial 
#    fractions xs, growth rates gs, lag times ls, and relative yields nus.
################################################################################
def CalcRelativeSaturationTime(xs, gs, ls, nus):
     precision = 1.e-6

     ind = numpy.argmin(ls)
     t = ls[ind]
     dt = 1/gs[ind]

     while abs(1 - CalcTotalResources(t, xs, gs, ls, nus)) > precision:
          while (1 - CalcTotalResources(t, xs, gs, ls, nus)) > 0:
               #print(("\tMoving ahead by " + str(dt) + " from time " + str(t) + ",").ljust(80), "Total resources:", CalcTotalResources(t, xs, gs, ls, nus))
               t += dt
          #print(("\t\tMoving backward by " + str(dt) + " from time " + str(t) + ",").ljust(80), "Total resources:", CalcTotalResources(t, xs, gs, ls, nus))
          t -= dt
          #t -= min(dt, t)
          dt *= 0.5

     return t


################################################################################
#    Calculates the exact selection coefficients for all pairs of strains in a 
#    multitype competition.
################################################################################
def CalcExactSij(xs, gs, ls, nus):
     tsat = CalcRelativeSaturationTime(xs, gs, ls, nus)
     fold_changes = [CalcFoldChange(tsat, gs[i], ls[i]) for i in range(len(xs))]

     sij = numpy.zeros( (len(xs), len(xs)) )
     for i in range(len(xs)):
          for j in range(i + 1, len(xs)):
               sij[i][j] = numpy.log(fold_changes[i]/fold_changes[j])
               sij[j][i] = -sij[i][j]
     return sij


################################################################################
#    Calculates the total selection coefficients on all strains using the matrix
#    of pairwise selection coefficients.
################################################################################
def CalcTotalSelection(xs, sij):
     numerators = numpy.array([sum(numpy.exp(-sij[i])*xs) - xs[i] for i in range(len(xs))])
     denominators = numpy.array([sum(xs) - xs[i] for i in range(len(xs))])
     return -numpy.log(numerators/denominators)


################################################################################
#    Evolves frequencies of an arbitrary number of strains up to rmax rounds of 
#    competition.
################################################################################
def CalcCompetitionTrajectory(xs, gs, ls, nus, rmax):
     freqs = [tuple(xs)]
     for r in range(rmax):
          tsat = CalcRelativeSaturationTime(freqs[-1], gs, ls, nus)
          Ns = [freqs[-1][i]*CalcFoldChange(tsat, gs[i], ls[i]) for i in range(len(gs))]
          Nsat = sum(Ns)
          freqs.append( tuple(N/Nsat for N in Ns) )
     return freqs


################################################################################
#    Calculates steady-state frequencies of an arbitrary number of strains.
################################################################################
def CalcSteadyStateFreqs(xs, gs, ls, nus):

     # Maximum number of competition rounds to try to converge
     rmax = 100000

     freqs_cur = [x for x in xs]
     for r in range(rmax):

          # Calculate frequencies for the next round
          tsat = CalcRelativeSaturationTime(freqs_cur, gs, ls, nus)
          Ns = [freqs_cur[i]*CalcFoldChange(tsat, gs[i], ls[i]) for i in range(len(gs))]
          Nsat = sum(Ns)
          freqs_next = [N/Nsat for N in Ns]

          # Calculate change in frequencies over this round and test
          distance = numpy.sqrt(sum((freqs_cur[i] - freqs_next[i])**2 for i in range(len(freqs_cur))))
          if distance < 1e-6:
               return r+1, freqs_next

          freqs_cur = [x for x in freqs_next]

     print("Steady-state calculation failed to converge!")
     return rmax, xs


################################################################################
#    Calculates exact fixation probability and time for two strains, where the
#    second strain (mutant) is assumed to start from a single cell.
################################################################################
def CalcExactM3Fixation(gamma, omega, nu1, nu2, N0):
     N = int(N0)

     # Calculate q's for all initial mutant frequencies
     qs = [0]
     for n in range(1, N):
          x = n/N
          s = CalcExactS(x, gamma, omega, nu1, nu2)  
          qs.append(x*numpy.exp(s)/(x*numpy.exp(s) + 1 - x))
     qs.append(1)
     #qs = [0] + [CalcExactSaturationFrequency(n/N, gamma, omega, nu1, nu2) for n in range(1, N)] + [1]

     # Set up transition submatrix between intermediate mutant frequencies
     A = numpy.zeros((N - 1, N - 1))
     for n1 in range(1, N):
          for n2 in range(1, N):
               A[n2 - 1][n1 - 1] = scipy.special.binom(N, n2)*( qs[n1]**n2 )*( (1 - qs[n1])**(N - n2) )
     H = numpy.identity(N - 1) - A

     # Set up transition submatrix of jumps from intermediate states to fixation
     B = numpy.array([qs[n]**N for n in range(1, N)])

     # Solve linear systems for fixation probabilities and mean times
     phis = numpy.linalg.solve(H.T, B)
     taus = numpy.linalg.solve(numpy.dot(H.T, H.T), B)

     # Return probability and time starting from a single mutant only
     return (phis[0], taus[0]/phis[0])


################################################################################
#    Calculates exact WF fixation probability and time for constant s by solving
#    the backward equation.
################################################################################
def CalcExactWFFixation(N, s):
     N = int(N)

     # Calculate q's for all mutant frequencies
     qs = [n*(1 + s)/(n*(1 + s) + N - n) for n in range(0, N + 1)]

     # Set up transition submatrix between intermediate mutant frequencies
     A = numpy.zeros((N - 1, N - 1))
     for n1 in range(1, N):
          for n2 in range(1, N):
               A[n2 - 1][n1 - 1] = scipy.special.binom(N, n2)*( qs[n1]**n2 )*( (1 - qs[n1])**(N - n2) )
     H = numpy.identity(N - 1) - A

     # Set up transition submatrix of jumps from intermediate states to fixation
     B = numpy.array([qs[n]**N for n in range(1, N)])

     # Solve linear systems for fixation probabilities and mean times
     phis = numpy.linalg.solve(H.T, B)
     taus = numpy.linalg.solve(numpy.dot(H.T, H.T), B)

     # Return probability and time starting from a single mutant only
     return (phis[0], taus[0]/phis[0])


################################################################################
#    This function determines if a set of three types forms a nontransitive 
#    loop.
################################################################################
def IsNontransitive(g1, g2, g3, l1, l2, l3, nu1, nu2, nu3):
     s21 = CalcApproxS(0.5, (g2 - g1)/g1, g1*(l2 - l1), nu1, nu2)
     s32 = CalcApproxS(0.5, (g3 - g2)/g2, g2*(l3 - l2), nu2, nu3)
     s13 = CalcApproxS(0.5, (g1 - g3)/g3, g3*(l1 - l3), nu3, nu1)
     return ((s21 > 0) and (s32 > 0) and (s13 > 0)) or ((s21 < 0) and (s32 < 0) and (s13 < 0))


################################################################################
#    Calculates the time at which the population growth rate reaches its 
#    maximum according to a convergence criterion.
################################################################################
def CalcConvergenceTime(xs, gs, ls):
     # The maximum growth rate
     max_growth_rate = max(gs)

     # Convergence criterion: how close the growth rate needs to be to the maximum before stopping
     precision = 1e-3

     t = 0
     while True:
          growth_rate = CalcPopGrowthRateExact(t, xs, gs, ls)
          if growth_rate == None:
               return None
          if abs(growth_rate - max_growth_rate)/max_growth_rate < precision:
               return t
          t += 1/max(gs)


################################################################################
#    Calculates components of pairwise selection coefficients on lineages when 
#    memory is short or resources are abundant.
################################################################################
def CalcPairwiseLineageSelectionComponentsShortMemory(gs, ls, gsteady, nmem):
     sij_lag = numpy.zeros((len(gs), len(ls)))
     sij_growth = numpy.zeros((len(gs), len(ls)))
     for i in range(len(gs)):
          for j in range(len(gs)):
               sij_lag[i][j] = -gsteady*(ls[i] - ls[j])
               sij_growth[i][j] = nmem*numpy.log(2)*gsteady*(gs[i] - gs[j])/gs[i]/gs[j]

     return sij_growth, sij_lag


################################################################################
#    Calculates pairwise selection coefficients on lineages when memory is short
#    or resources are abundant.
################################################################################
def CalcPairwiseLineageSelectionShortMemory(gs, ls, gsteady, nmem):
     sij_lag, sij_growth = CalcPairwiseLineageSelectionComponentsShortMemory(gs, ls, gsteady, nmem)
     return sij_growth + sij_lag


################################################################################
################################################################################
#################### APPROXIMATE FUNCTIONS FOR GROWTH MODEL ####################
################################################################################
################################################################################


################################################################################
#    Calculates components of the approximate selection coefficients for all 
#    pairs of strains in a multitype competition.
################################################################################
def CalcApproxSijComponents(xs, gs, ls, nus):
     # Convert growth rates to growth times
     ts = [1/g for g in gs]

     # Relative carrying capacity (effective yield)
     K = 1/sum(xs[i]/nus[i] for i in range(len(xs)))

     # Effective growth time scale
     T = 1/sum(xs[i]/nus[i]/ts[i] for i in range(len(xs)))/K

     sij_growth = numpy.zeros( (len(xs), len(xs)) )
     sij_lag = numpy.zeros( (len(xs), len(xs)) )
     sij_coupling = numpy.zeros( (len(xs), len(xs)) )
     for i in range(len(xs)):
          for j in range(i + 1, len(xs)):
               prefactor = T/ts[i]/ts[j]

               sij_growth[i][j] = -prefactor*(ts[i] - ts[j])*numpy.log(K)
               sij_growth[j][i] = -sij_growth[i][j]

               sij_lag[i][j] = -prefactor*(ls[i] - ls[j])
               sij_lag[j][i] = -sij_lag[i][j]

               sij_coupling[i][j] = -prefactor*K*sum(((ts[i] - ts[k])*(ls[k] - ls[j]) - (ls[i] - ls[k])*(ts[k] - ts[j]))*xs[k]/nus[k]/ts[k] for k in range(len(xs)))
               sij_coupling[j][i] = -sij_coupling[i][j]

     return sij_growth, sij_lag, sij_coupling


################################################################################
#    Calculates the approximate selection coefficients for all pairs of strains 
#    in a multitype competition.
################################################################################
def CalcApproxSij(xs, gs, ls, nus):
     sij_growth, sij_lag, sij_coupling = CalcApproxSijComponents(xs, gs, ls, nus)
     return sij_growth + sij_lag + sij_coupling


################################################################################
#    Calculates components of the approximate selection coefficients just 
#    between strains i and j in a multitype competition.
################################################################################
def CalcApproxSijComponentsMultitype(i, j, xs, gs, ls, nus):
     # Convert growth rates to growth times
     ts = [1/g for g in gs]

     # Relative carrying capacity (effective yield)
     K = 1/sum(xs[i]/nus[i] for i in range(len(xs)))

     # Effective growth time scale
     T = 1/sum(xs[i]/nus[i]/ts[i] for i in range(len(xs)))/K

     prefactor = T/ts[i]/ts[j]

     sij_growth = -prefactor*(ts[i] - ts[j])*numpy.log(K)
     sij_lag = -prefactor*(ls[i] - ls[j])
     sij_coupling = -prefactor*K*sum(((ts[i] - ts[k])*(ls[k] - ls[j]) - (ls[i] - ls[k])*(ts[k] - ts[j]))*xs[k]/nus[k]/ts[k] for k in range(len(xs)))

     return sij_growth, sij_lag, sij_coupling


################################################################################
#    Calculates the Jacobian of the selection coefficient at a frequency point.
#    The selection coefficients are all assumed to be relative to the first
#    strain.
################################################################################
def CalcApproxSijDerivativesMultitype(xs, gs, ls, nus):
     ts = [1/g for g in gs]

     sij = CalcApproxSMultitype(xs, gs, ls, nus)

     H = 1/sum(xs[i]/nus[i] for i in range(len(xs)))
     T = 1/sum(xs[i]/nus[i]/ts[i] for i in range(len(xs)))/H

     Gij = numpy.zeros( (len(xs), len(xs)) )
     for i in range(len(xs)):
          for j in range(len(xs)):
               term1 = H*(1 - T/ts[j])*sij[i][0]/nus[j]
               term2_prefactor = H*T/nus[j]/ts[i]/ts[0]
               term2_sign = ts[i] - ts[0] - H*sum((-(ts[i] - ts[0])*(ls[k] - ls[0]) + (ls[i] - ls[0])*(ts[k] - ts[0]))*xs[k]/nus[k]/ts[k] for k in range(len(xs))) + (-(ts[i] - ts[0])*(ls[j] - ls[0]) + (ls[i] - ls[0])*(ts[j] - ts[0]))/ts[j]
               Gij[i][j] = term1 + term2_prefactor*term2_sign

     return Gij


################################################################################
#    Calculates the relative variation in selection: ratio of selection 
#    coefficient range to value at x = 1/2.
################################################################################
def CalcApproxSVariation(gamma, omega, nu1, nu2):
     total_range = gamma*(numpy.log(nu2)/(1 + gamma) - numpy.log(nu1) + omega)
     return abs(total_range/CalcApproxSBinary(0.5, gamma, omega, nu1, nu2))


################################################################################
#    Calculates Nc, the critical population size.
################################################################################
def CalcNc(gamma, omega):
     return numpy.exp(omega*(1 + 1/gamma))


################################################################################
#    This function calculates eta (relative selection on lag versus growth).
#    It is negative if there is a tradeoff between growth and lag, and positive
#    if not.
################################################################################
def CalcEta(x, gamma, omega, nu1, nu2):
     return -omega*(1 + 1/gamma)/numpy.log(0.5*HarmonicMean(nu1/(1 - x), nu2/x))


################################################################################
#    Calculates the fixed point frequency of a mutant.
################################################################################
def CalcMutantFixedPoint(gamma, omega, nu1, nu2):
     Nc = CalcNc(gamma, omega)
     return (nu1/Nc - 1)/(nu1/nu2 - 1)


################################################################################
#    Calculates the maximum entropy frequencies with coexistence.
################################################################################
def CalcMaxEntFreq(nus, c):
     precision = 1.e-8
     max_iterations = 10000

     # Function to calculate mean
     mean = lambda b: sum(numpy.exp(-b/nu)/nu for nu in nus)/sum(numpy.exp(-b/nu) for nu in nus)

     # Initial value for beta
     beta = 0
     if mean(beta) > numpy.exp(-c):
          sign = 1
     else:
          sign = -1

     # Initial increment
     dbeta = sign*numpy.mean(nus)

     iterations = 0     
     while abs(1 - mean(beta)/numpy.exp(-c)) > precision:
          while sign*(mean(beta) - numpy.exp(-c)) > 0:
               iterations += 1
               if iterations > max_iterations:
                    print("MaxEnt calculation not converging.")
                    print(a, nus)
                    return False
               beta += dbeta
          beta -= dbeta
          dbeta *= 0.5
          
     # Calculate normalization
     Z = sum(numpy.exp(-beta/nu) for nu in nus)

     return [numpy.exp(-beta/nu)/Z for nu in nus]


################################################################################
#    Calculates the mean approximate change in frequency of the mutant over one
#    round of competition.
################################################################################
def CalcApproxM1(x, gamma, omega, nu1, nu2):
     q = CalcApproxSaturationFrequency(x, gamma, omega, nu1, nu2)
     return q - x


################################################################################
#    Calculates the variance in the change of frequency of the mutant over one
#    round of competition.
################################################################################
def CalcApproxM2(x, gamma, omega, nu1, nu2, N0):
     q = CalcApproxSaturationFrequency(x, gamma, omega, nu1, nu2)
     return q*(1 - q)/N0 #+ (q - x)**2


################################################################################
#    Calculates the effective potential, i.e., the negative integral of the 
#    force (selection coefficient).
################################################################################
def CalcV(x, gamma, omega, nu1, nu2, N0):
     func = lambda x, gamma, omega, nu1, nu2, N0: -2*N0*CalcApproxSBinary(x, gamma, omega, nu1, nu2)
     #func = lambda x, gamma, omega, nu1, nu2, N0: -2*CalcApproxM1(x, gamma, omega, nu1, nu2)/CalcApproxM2(x, gamma, omega, nu1, nu2, N0)
     result, err = scipy.integrate.quad(func, 0, x, args=(gamma, omega, nu1, nu2, N0))
     return result


################################################################################
#    Calculates the fixation probability using the diffusion approximation.
################################################################################
def CalcDiffusionPhi(x, gamma, omega, nu1, nu2, N0):
     func = lambda y, gamma, omega, nu1, nu2, N0: numpy.exp(CalcV(y, gamma, omega, nu1, nu2, N0))
     numerator, err = scipy.integrate.quad(func, 0, x, args=(gamma, omega, nu1, nu2, N0))
     denominator, err = scipy.integrate.quad(func, 0, 1, args=(gamma, omega, nu1, nu2, N0))
     return numerator/denominator


################################################################################
#    Calculates the fixation time using the diffusion approximation.
################################################################################
def CalcDiffusionTau(x, gamma, omega, nu1, nu2, N0):
     func = lambda x, gamma, omega, nu1, nu2, N0: numpy.exp(CalcV(x, gamma, omega, nu1, nu2, N0))
     c, err = scipy.integrate.quad(func, 0, 1, args=(gamma, omega, nu1, nu2, N0))

     psi = lambda x, gamma, omega, nu1, nu2, N0: 2*c/CalcApproxM2(x, gamma, omega, nu1, nu2, N0)/func(x, gamma, omega, nu1, nu2, N0)
     phi_numerator = lambda x, gamma, omega, nu1, nu2, N0: scipy.integrate.quad(func, 0, x, args=(gamma, omega, nu1, nu2, N0))[0]
     phi_denominator = scipy.integrate.quad(func, 0, 1, args=(gamma, omega, nu1, nu2, N0))[0]
     phi = CalcDiffusionPhi(x, gamma, omega, nu1, nu2, N0)

     integrand1 = lambda x, gamma, omega, nu1, nu2, N0: psi(x, gamma, omega, nu1, nu2, N0)*(1 - phi_numerator(x, gamma, omega, nu1, nu2, N0)/phi_denominator)*phi_numerator(x, gamma, omega, nu1, nu2, N0)/phi_denominator
     integrand2 = lambda x, gamma, omega, nu1, nu2, N0: psi(x, gamma, omega, nu1, nu2, N0)*(phi_numerator(x, gamma, omega, nu1, nu2, N0)/phi_denominator)**2
     
     return scipy.integrate.quad(integrand1, x, 1, args=(gamma, omega, nu1, nu2, N0))[0] + ((1 - phi)/phi)*scipy.integrate.quad(integrand2, 0, x, args=(gamma, omega, nu1, nu2, N0))[0]


################################################################################
#    Calculates the mean absorption (fixation or extinction) time using the 
#    diffusion approximation.
################################################################################
def CalcDiffusionAlpha(x, gamma, omega, nu1, nu2, N0):
     func = lambda x, gamma, omega, nu1, nu2, N0: numpy.exp(CalcV(x, gamma, omega, nu1, nu2, N0))
     c, err = scipy.integrate.quad(func, 0, 1, args=(gamma, omega, nu1, nu2, N0))

     psi = lambda x, gamma, omega, nu1, nu2, N0: 2*c/CalcApproxM2(x, gamma, omega, nu1, nu2, N0)/func(x, gamma, omega, nu1, nu2, N0)
     phi_numerator = lambda x, gamma, omega, nu1, nu2, N0: scipy.integrate.quad(func, 0, x, args=(gamma, omega, nu1, nu2, N0))[0]
     phi_denominator = scipy.integrate.quad(func, 0, 1, args=(gamma, omega, nu1, nu2, N0))[0]
     phi = CalcDiffusionPhi(x, gamma, omega, nu1, nu2, N0)

     integrand1 = lambda x, gamma, omega, nu1, nu2, N0: psi(x, gamma, omega, nu1, nu2, N0)*(1 - phi_numerator(x, gamma, omega, nu1, nu2, N0)/phi_denominator)
     integrand2 = lambda x, gamma, omega, nu1, nu2, N0: psi(x, gamma, omega, nu1, nu2, N0)*(phi_numerator(x, gamma, omega, nu1, nu2, N0)/phi_denominator)

     return phi*scipy.integrate.quad(integrand1, x, 1, args=(gamma, omega, nu1, nu2, N0))[0] + (1 - phi)*scipy.integrate.quad(integrand2, 0, x, args=(gamma, omega, nu1, nu2, N0))[0]


################################################################################
#    Calculates Kimura's formula (diffusion approximation for the fixation 
#    probability) for a mutant with selection coefficient s and effective 
#    population size N.
################################################################################
def CalcKimuraPhi(x, N, s):

     # If the mutant is exactly neutral, then just return the frequency
     if s == 0:
          return x
     else:
          # First try to calculate explicit formula
          try:
               return (1 - numpy.exp(-2*N*s*x))/(1 - numpy.exp(-2*N*s))
          # If there is an overflow in the exponential in the denominator, it is 
          # because the mutation is too deleterious relative to the population 
          # size.  In that case the fixation probability should be taken as zero.
          except RuntimeWarning:
               return 0


################################################################################
#    Calculates Kimura's formula (diffusion approximation for the mean fixation 
#    time) for a mutant with selection coefficient s and effective 
#    population size N.
################################################################################
def CalcKimuraTau(x, N, s):
     psi = lambda x, N, s: (1 - numpy.exp(-2*N*s))*numpy.exp(2*N*s*x)/(s*x*(1 - x))
     phi = CalcKimuraPhi(x, N, s)

     integrand1 = lambda x, N, s: psi(x, N, s)*(1 - CalcKimuraPhi(x, N, s))*CalcKimuraPhi(x, N, s)
     integrand2 = lambda x, N, s: psi(x, N, s)*CalcKimuraPhi(x, N, s)**2
     
     return scipy.integrate.quad(integrand1, x, 1, args=(N, s))[0] + ((1 - phi)/phi)*scipy.integrate.quad(integrand2, 0, x, args=(N, s))[0]


################################################################################
#    Calculates the effective population growth rate.
################################################################################
def CalcEffGrowthRate(xs, gs, ls):
     return sum(xs*gs*numpy.exp(-gs*ls))/sum(xs*numpy.exp(-gs*ls))


################################################################################
#    Calculates the effective variance of growth rates.
################################################################################
def CalcEffGrowthVar(xs, gs, ls):
     geff = CalcEffGrowthRate(xs, gs, ls)
     g2eff = sum(xs*gs**2*numpy.exp(-gs*ls))/sum(xs*numpy.exp(-gs*ls))
     return (g2eff - geff**2)


################################################################################
#    Calculates the effective Fano factor of growth rates.
################################################################################
def CalcEffGrowthFano(xs, gs, ls):
     return CalcEffGrowthVar(xs, gs, ls)/CalcEffGrowthRate(xs, gs, ls)


################################################################################
#    Calculates the effective coefficient of variation of growth rates.
################################################################################
def CalcEffGrowthCV(xs, gs, ls):
     return numpy.sqrt(CalcEffGrowthVar(xs, gs, ls))/CalcEffGrowthRate(xs, gs, ls)


################################################################################
#    Calculates the effective population lag time.
################################################################################
def CalcEffLagTime(xs, gs, ls):
     return -numpy.log(sum(xs*numpy.exp(-gs*ls)))/CalcEffGrowthRate(xs, gs, ls)


################################################################################
#    Calculates the probability density of a lineage acquiring frequency x after
#    growth in a population of N0 lineages and lag times have a gamma 
#    distribution with mean l_mean and standard deviation l_sd.
################################################################################
def CalcProbFreqGammaLagTimes(x, g, l_mean, l_sd, N0):
     cv = l_sd/l_mean
     xmax = (1 + g*l_mean*cv**2)**(1/cv**2)/N0
     if x > xmax: return 0

     prefactor = 1/scipy.special.gamma(1/cv**2)/(g*l_mean*cv**2)**(1/cv**2)
     xdependence = (x/xmax)**(1/g/l_mean/cv**2)*numpy.log(xmax/x)**(1/cv**2 - 1)/x
     return prefactor*xdependence


################################################################################
################################################################################
############################ DEPRECATED FUNCTIONS ##############################
################################################################################
################################################################################


################################################################################
#    Calculates the exact selection coefficient of a mutant relative to 
#    wild-type, where the mutant has initial frequency x, relative growth rate
#    gamma, relative lag time omega, and relative yields nu1 = RY1/N0 (wild-
#    type) and nu2 = RY2/N0 (mutant).
################################################################################
#def CalcExactSBinary(x, gamma, omega, nu1, nu2):
#     tsat = CalcRelativeSaturationTime([1 - x, x], [1, 1 + gamma], [0, omega], [nu1, nu2])
#     s = numpy.log(CalcFoldChange(tsat, 1 + gamma, omega)/CalcFoldChange(tsat, 1, 0))
#     return s


################################################################################
#    Calculates the exact frequency of the mutant at saturation.
################################################################################
#def CalcExactSaturationFrequency(x, gamma, omega, nu1, nu2):
#     s = CalcExactS(x, gamma, omega, nu1, nu2)
#     return x*numpy.exp(s)/(x*numpy.exp(s) + 1 - x)


################################################################################
#    Evolves a mutant frequency over rmax rounds of competition.
################################################################################
#def CalcExactMutantFreqTrajectory(x, gamma, omega, nu1, nu2, rmax):
#     freqs = [x]
#     for r in range(rmax):
#          x = CalcExactSaturationFrequency(x, gamma, omega, nu1, nu2)
#          freqs.append(x)
#     return freqs


################################################################################
#    This function returns the outcome of a triple competition, assuming the 
#    total initial population size is N0, split evenly into the three types.  It
#    returns the fractions of each type at saturation.
################################################################################
#def CalcTripletCompetition(x1, x2, x3, g1, g2, g3, l1, l2, l3, nu1, nu2, nu3):
#     tsat = CalcRelativeSaturationTime([x1, x2, x3], [g1, g2, g3], [l1, l2, l3], [nu1, nu2, nu3], 1)
#     x1 = CalcN(tsat, x1, g1, l1)
#     x2 = CalcN(tsat, x2, g2, l2)
#     x3 = CalcN(tsat, x3, g3, l3)
#     xtotal = x1 + x2 + x3
#     return x1/xtotal, x2/xtotal, x3/xtotal


################################################################################
#    Evolves frequencies of three strains over rmax rounds of competition.
################################################################################
#def CalcTripletCompetitionTrajectory(x1, x2, x3, g1, g2, g3, l1, l2, l3, nu1, nu2, nu3, rmax):
#     freqs = [(x1, x2, x3)]
#     for r in range(rmax):
#          x1, x2, x3 = CalcTripletCompetition(x1, x2, x3, g1, g2, g3, l1, l2, l3, nu1, nu2, nu3)
#          freqs.append((x1, x2, x3))
#     return freqs


################################################################################
#    Calculates the approximate selection coefficient of a mutant relative to 
#    wild-type, where the mutant has initial frequency x, relative growth rate
#    gamma, relative lag time omega, and relative yields nu1 = RY1/N0 (wild-
#    type) and nu2 = RY2/N0 (mutant).
################################################################################
#def CalcApproxSBinary(x, gamma, omega, nu1, nu2):
#     time_scale = ( (1 - x)/nu1 + x/nu2 )/( (1 - x)/nu1 + (1 + gamma)*x/nu2 )
#     sign = gamma*numpy.log(0.5*HarmonicMean([nu1/(1 - x), nu2/x])) - omega*(1 + gamma)
#     return time_scale*sign


################################################################################
#    Calculates the approximate growth selection coefficient of a mutant 
#    relative to wild-type, where the mutant has initial frequency x, relative 
#    growth rate gamma, relative lag time omega, and relative yields nu1 = 
#    RY1/N0 (wild-type) and nu2 = RY2/N0 (mutant).
################################################################################
#def CalcApproxSGrowthBinary(x, gamma, omega, nu1, nu2):
#     time_scale = ( (1 - x)/nu1 + x/nu2 )/( (1 - x)/nu1 + (1 + gamma)*x/nu2 )
#     sign = gamma*numpy.log(0.5*HarmonicMean([nu1/(1 - x), nu2/x]))
#     return time_scale*sign


################################################################################
#    Calculates the approximate growth selection coefficient of a mutant 
#    relative to wild-type, where the mutant has initial frequency x, relative 
#    growth rate gamma, relative lag time omega, and relative yields nu1 = 
#    RY1/N0 (wild-type) and nu2 = RY2/N0 (mutant).
################################################################################
#def CalcApproxSLagBinary(x, gamma, omega, nu1, nu2):
#     time_scale = ( (1 - x)/nu1 + x/nu2 )/( (1 - x)/nu1 + (1 + gamma)*x/nu2 )
#     sign = - omega*(1 + gamma)
#     return time_scale*sign


################################################################################
#    Calculates all approximate selection coefficients for a multitype 
#    competition by numerically solving the linear system.
################################################################################
#def CalcApproxSMultitype_System(xs, gammas, omegas, nus):
#     #gammas = numpy.array([(g - gs[0])/gs[0] for g in gs])
#     #omegas = numpy.array([gs[0]*(l - ls[0]) for l in ls])
#     #N0 = sum(N0s)
#     #xs = numpy.array([n/N0 for n in N0s])
#     xs = numpy.array(xs)
#     gammas = numpy.array(gammas)
#     omegas = numpy.array(omegas)
#     nus = numpy.array(nus)
#
#     # Total number of types
#     M = len(xs)
#
#     # Harmonic mean
#     H = M/sum(xs/nus)
#
#     # Matrix
#     A = numpy.zeros((M, M))
#     for i in range(M):
#          for j in range(M):
#               A[i][j] = (H/M)*gammas[i]*xs[j]/nus[j] + int(i == j)
#
#     # Right-hand side of linear system
#     B = gammas*numpy.log(H/M) - omegas*(1 + gammas)
#
#     # Solve for selection coefficients
#     ss = numpy.linalg.solve(A, B)
#     return ss


################################################################################
#    Calculates the derivative of the approximate selection coefficient with 
#    respect to frequency x.
################################################################################
#def CalcApproxSDerivativesBinary(x, gamma, omega, nu1, nu2):
#     s = CalcApproxS(x, gamma, omega, nu1, nu2)
#     numerator = gamma*( nu2 - nu1 - s/((1 - x)/nu1 + x/nu2) )
#     denominator = nu1*nu2*( (1 - x)/nu1 + (1 + gamma)*x/nu2 )
#     return numerator/denominator


################################################################################
#    Calculates the stability (derivative of selection coefficient) at a fixed
#    point.
################################################################################
#def CalcFixedPointStability(gamma, omega, nu1, nu2):
#     xtilde = CalcMutantFixedPoint(gamma, omega, nu1, nu2)
#     numerator = gamma*(nu2/nu1 - 1)
#     denominator = xtilde*(1 + gamma) + (1 - xtilde)*nu2/nu1
#     return numerator/denominator


################################################################################
#    Calculates the approximate frequency of the mutant at saturation.
################################################################################
#def CalcApproxSaturationFrequency(x, gamma, omega, nu1, nu2):
#     s = CalcApproxSBinary(x, gamma, omega, nu1, nu2)
#     return x*numpy.exp(s)/(x*numpy.exp(s) + 1 - x)


################################################################################
#    This function calculates the dimensionless area of a triangle of points in
#    g-lambda space.
################################################################################
#def CalcArea(g1, g2, g3, l1, l2, l3):
#     return 0.5*abs(g1*l2 + g2*l3 + g3*l1 - g2*l1 - g3*l2 - g1*l3)


################################################################################
#    This function calculates the direction of the normal to a triangle of 
#    points in g-lambda-Nmax space.
################################################################################
#def CalcNormal(g1, g2, g3, l1, l2, l3, nu1, nu2, nu3):
#     v = numpy.array([(g2 - g1)/g1, (l2 - l1)*g1, nu2 - nu1])
#     w = numpy.array([(g3 - g1)/g1, (l3 - l1)*g1, nu3 - nu1])
#     n = numpy.cross(v, w)
#     #return Sign(n[0]), Sign(n[1]), Sign(n[2])
#     return n


################################################################################
#    Given two strains, determine the winning strategy.
#         "Champion":    Better growth and better lag
#         "Crammer":     Better growth but worse lag, and better efficiency
#         "Slash&Burn":  Better growth but worse lag, and worse efficiency
#         "EarlyBird":   Worse growth but better lag, and better efficiency
#         "Hoarder":     Worse growth but better lag, and worse efficiency
################################################################################
#def GetStrategy(gamma, omega, nu1, nu2):
#
#     # Calculate selection coefficient of 2 over 1
#     s21 = CalcApproxS(0.5, gamma, omega, nu1, nu2)
#
#     # Set winner parameters to whichever strain is overall better
#     if s21 < 0:
#          gamma_original = gamma
#          omega_original = omega
#          nu1_original = nu1
#          nu2_original = nu2
#          gamma = -gamma_original/(1 + gamma_original)
#          omega = -omega_original*(1 + gamma_original)
#          nu1 = nu2_original
#          nu2 = nu1_original
#
#     # Calculate eta for winner
#     eta = CalcEta(0.5, gamma, omega, nu1, nu2)
#
#     # If eta > 0, there is no tradeoff and the winner is wholly better
#     if eta > 0:
#          return "Champion"
#     
#     # If eta < 0, there is a tradeoff between growth and lag
#     if (gamma > 0) and (nu2 > nu1):
#          return "Crammer"
#     if (gamma > 0) and (nu2 < nu1):
#          return "Slash&Burn"
#     if (gamma < 0) and (nu2 > nu1):
#          return "EarlyBird"
#     if (gamma < 0) and (nu2 < nu1):
#          return "Hoarder"
#
#     return "Unknown"


################################################################################
#    This function returns the list of all distinct strategies around a 
#    nontransitive loop.
################################################################################
#def GetAllDistinctStrategies():
#     all_strategy_triplets = set([])
#
#     # Loop over all possible triplets of strategies
#     for triplet in product(["Champion", "Crammer", "Slash&Burn", "EarlyBird", "Hoarder"], repeat=3):
#          s1, s2, s3 = triplet
#
#          # If no cyclic permutation of triplet is already in set, reorder and add
#          if ((s1, s2, s3) not in all_strategy_triplets) and ((s2, s3, s1) not in all_strategy_triplets) and ((s3, s1, s2) not in all_strategy_triplets):
#               # Put triplet of strategies in a standard order for comparison     
#               first = numpy.argmin([s1, s2, s3])
#               if first == 0:
#                    all_strategy_triplets.add( (s1, s2, s3) )
#               elif first == 1:
#                    all_strategy_triplets.add( (s2, s3, s1) )
#               else:
#                    all_strategy_triplets.add( (s3, s1, s2) )
#
#     return list(all_strategy_triplets)


################################################################################
#    This function returns the triplet of strategies.
################################################################################
#def GetAllStrategies(g1, g2, g3, l1, l2, l3, nu1, nu2, nu3):
#
#     strategy21 = GetStrategy((g2 - g1)/g1, g1*(l2 - l1), nu1, nu2)
#     strategy32 = GetStrategy((g3 - g2)/g2, g2*(l3 - l2), nu2, nu3)
#     strategy13 = GetStrategy((g1 - g3)/g3, g3*(l1 - l3), nu3, nu1)
#
#     # Put triplet of strategies in a standard order for comparison     
#     first = numpy.argmin([strategy21, strategy32, strategy13])
#     if first == 0:
#          return (strategy21, strategy32, strategy13)
#     elif first == 1:
#          return (strategy32, strategy13, strategy21)
#     else:
#          return (strategy13, strategy21, strategy32)


################################################################################
#    This returns a triplet of types in a standard order: the first type always
#    has the lowest growth rate, and the order goes with positive selection 
#    (1 -> 2 -> 3 -> 1).
################################################################################
#def GetStandardOrder(g1, g2, g3, l1, l2, l3, nu1, nu2, nu3):
#
#     # Sort types according to growth rate
#     traits = [(g1, l1, nu1), (g2, l2, nu2), (g3, l3, nu3)]
#     traits.sort(key=lambda x: x[0])
#     g1, l1, nu1 = traits[0]
#     g2, l2, nu2 = traits[1]
#     g3, l3, nu3 = traits[2]
#
#     # Determine sign of selection around loop, and reverse order if necessary
#     s21 = CalcApproxS(0.5, (g2 - g1)/g1, g1*(l2 - l1), nu1, nu2)
#     if s21 > 0:
#          return g1, g2, g3, l1, l2, l3, nu1, nu2, nu3
#     else:
#          return g1, g3, g2, l1, l3, l2, nu1, nu3, nu2
