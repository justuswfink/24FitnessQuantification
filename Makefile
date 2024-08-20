# Makefile for running notebooks 


# Do we need Phony targets with pandoc?
.PHONY: all  upload

all: 

upload:
	rsync -avp -u figures/ ~/polybox/21FitnessQuantifiication/figures/
	rsync -avp -u output/ ~/polybox/21FitnessQuantifiication/output/

download:
	rsync -avp -u ~/polybox/21FitnessQuantifiication/output/ output/

setup:
	mkdir logs


%.papermill:
	papermill $*.ipynb logs/$*.ipynb

absolute_fitness: 
	# when all traits vary, we run two frequencies
	parallel papermill 34_plot_absolute_fitness_predictions.ipynb \
		logs/34_plot_absolute_fitness_predictions_all_traits_vary_{1}.ipynb\
		-p NO_CURVES None -p DIST all_traits_vary  -p INITIAL_FREQ {1} \
		::: 0.01 0.5
	# when some of trait variation is reduced, only run frequency x = 0.5
	parallel papermill 34_plot_absolute_fitness_predictions.ipynb \
		logs/34_plot_absolute_fitness_predictions_{1}_0.5.ipynb\
		-p NO_CURVES None -p DIST {1}  -p INITIAL_FREQ 0.5 \
		::: no_yield_variation no_deleterious_yield no_gmax_variation\
		no_lag_variation only_gmax_varies only_lag_varies
	# when all traits vary, we run three cutoff times
	parallel papermill 34_plot_absolute_fitness_predictions.ipynb \
		logs/34_plot_absolute_fitness_predictions_all_traits_vary_0.5_cutoff_{1}.ipynb\
		-p NO_CURVES None -p DIST all_traits_vary  -p INITIAL_FREQ 0.5\
		-p AUC_CUTOFF_TIME {1} ::: 13 16 24

disranking: 
	# when all traits vary, we run two frequencies
	parallel papermill 33_plot_selection_timescales_warringer2003.ipynb \
		logs/33_plot_selection_timescales_warringer2003_all_traits_vary_{1}.ipynb\
		-p NO_CURVES None -p DIST all_traits_vary  -p INITIAL_FREQ {1} \
		::: 0.01 0.5
	# when some of trait variation is reduced, only run frequency x = 0.5
	parallel papermill 33_plot_selection_timescales_warringer2003.ipynb \
		logs/33_plot_selection_timescales_warringer2003_{1}_0.5.ipynb\
		-p NO_CURVES None -p DIST {1}  -p INITIAL_FREQ 0.5 \
		::: no_yield_variation no_deleterious_yield no_gmax_variation\
		no_lag_variation only_gmax_varies only_lag_varies

trait_correlation: 
	# when all traits vary, we run two frequencies
	papermill 32_plot_trait_correlations_warringer2003.ipynb \
	logs/32_plot_trait_correlations_warringer2003_all_traits_vary.ipynb\
		-p NO_CURVES None -p DIST all_traits_vary 
	# when some of trait variation is reduced, only run frequency x = 0.5
	parallel papermill 32_plot_trait_correlations_warringer2003.ipynb \
		logs/32_plot_trait_correlations_warringer2003_{1}.ipynb\
		-p NO_CURVES None -p DIST {1} \
		::: no_yield_variation no_deleterious_yield no_gmax_variation\
		no_lag_variation only_gmax_varies only_lag_varies
simulations: 
	# when all traits vary, we run two frequencies
	parallel papermill simulations_relative_fitness_warringer2003.ipynb \
		logs/simulations_relative_fitness_warringer2003_all_traits_vary_{1}.ipynb\
		-p NO_CURVES None -p DIST all_traits_vary  -p INITIAL_FREQ {1} \
		::: 0.01 0.5
	# when some of trait variation is reduced, only run frequency x = 0.5
	parallel papermill simulations_relative_fitness_warringer2003.ipynb \
		logs/simulations_relative_fitness_warringer2003_{1}_0.5.ipynb\
		-p NO_CURVES None -p DIST {1}  -p INITIAL_FREQ 0.5 \
		::: no_yield_variation no_deleterious_yield no_gmax_variation\
		no_lag_variation only_gmax_varies only_lag_varies

plots: 
	# when all traits vary, we run two frequencies
	papermill plot_correlation_relative_fitness_warringer2003.ipynb - -p DIST all_traits_vary  -p INITIAL_FREQ 0.01 2>&1>/dev/null 
	papermill plot_correlation_relative_fitness_warringer2003.ipynb - -p DIST all_traits_vary  -p INITIAL_FREQ 0.5 2>&1>/dev/null
	# when some of trait variation is reduced, only run frequency x = 0.5
	papermill plot_correlation_relative_fitness_warringer2003.ipynb - -p DIST no_yield_variation  -p INITIAL_FREQ 0.5 2>&1>/dev/null
	papermill plot_correlation_relative_fitness_warringer2003.ipynb - -p DIST no_deleterious_yield  -p INITIAL_FREQ 0.5 2>&1>/dev/null
	papermill plot_correlation_relative_fitness_warringer2003.ipynb - -p DIST no_gmax_variation  -p INITIAL_FREQ 0.5 2>&1>/dev/null
	papermill plot_correlation_relative_fitness_warringer2003.ipynb - -p DIST no_lag_variation  -p INITIAL_FREQ 0.5 2>&1>/dev/null
	papermill plot_correlation_relative_fitness_warringer2003.ipynb - -p DIST only_gmax_varies  -p INITIAL_FREQ 0.5 2>&1>/dev/null
	papermill plot_correlation_relative_fitness_warringer2003.ipynb - -p DIST only_lag_varies  -p INITIAL_FREQ 0.5 2>&1>/dev/null

warringer2003: reload_warringer2003_fits.papermill
	papermill simulations_relative_fitness_warringer2003.ipynb logs/simulations_relative_fitness_warringer2003.ipynb -p NO_CURVES None
	papermill simulations_absolute_fitness_proxies_warringer2003.ipynb logs/simulations_absolute_fitness_proxies_warringer2003.ipynb -p NO_CURVES None

relative_fitness: simulations_relative_fitness_synthetic_correlated.ipynb
	papermill  $< logs/$< -p SUFFIX_DATASET uniform_balanced -p NO_CURVES 1000
	papermill  $< logs/$< -p SUFFIX_DATASET uniform_widelag -p NO_CURVES 1000
	papermill  $< logs/$< -p SUFFIX_DATASET uniform_widegmax -p NO_CURVES 1000
	papermill  $< logs/$< -p SUFFIX_DATASET pleiotropic_handcrafted -p NO_CURVES 100
	papermill  $< logs/$< -p SUFFIX_DATASET Gaussian_balanced -p NO_CURVES 1000



%.compressed: %.pdf
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$@.pdf $<

handover:
	mkdir -p handover
	rsync -avp latex/{article.tex,Makefile,mycommands.sty,literature.bib} handover/
	rsync -avp latex/figures_cropped.pdf handover/
	# rsync -avp latex/figures/pdfcrop.sh handover/figures/
	cd handover; make setup
	# cd handover; make download crop
	cd handover; make article 
	cd handover; make clean


%.html: %.ipynb
	jupyter nbconvert $< --to html

%.pdf: %.ipynb
	jupyter nbconvert $< --to pdf
	mv $@ build/$@

reset:
	rm -r ./figures/*


# set default target
%.set_default: % 
	 # "We have checked that the target builds. Now we make it the default 'all' target."
	 sed -i "s/^all:.*/all: $</g" Makefile 
