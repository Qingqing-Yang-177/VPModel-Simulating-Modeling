## Simulation and Modeling of Variable Precision Model
- The Modelling the Working memory resource allocation

This repository is a fork of the original repository from Yoo Aspen (https://github.com/aspenyoo/WM_resource_allocation). The original parent repository is accompanied from Yoo AH's paper paper *Strategic allocation of working memory resource* by Yoo, Klyszejko, Curtis, & Ma (2018, Sci Reports). [\[link\]](https://www.nature.com/articles/s41598-018-34282-1.pdf)

I created this repository as the result of my simulation and modeling of the variable precision model based on Yoo's codes. I also write a documentation of a introduction of variable precision model, and my simulation and modeling practices as a review of the VP model, which i could share with u if u contact me by qy775@nyu.edu

---------------------

#### My self contributions are:
- **simulate&model_QY.m:** the codes for simulation and modeling practice
  - Simulate data from evenly allocation VP model parameters, with diff params combinations.
  - Simulate from proportional allocation VP model parameters, with fixed Jbar or fixed tau.
  - Fit real data with negative Log Likelihood calculation.
  - Recover the parameter combination (theta2) from the simulated data (theta1)
- **Proportional_VP_single_simulator_QY:** the function for simulate a error data by Proportional VP model with a params combination [Jbar, tau] 
- **Proportional_calc_nll_QY.m:** **（revision needed）** the function for calculate the neg log-likelihood based on a data and Propor_VP model with a params combination [Jbar, tau], works for 1 by 3 cells while not 1 by 1 cell.
- **Proportional_fitparams_QY.m:** **（revision needed）** the function for get the best fitting param combination from the data, by certain runs of calculations, works for 1 by 3 cells while not 1 by 1 cell.
- **qy_modelling results:** contains results of my model simulation, parameter recovery, model comparison

------------------------


#### A brief description of the orginal repository organization (details are found within each file):
- **data/priority:** contains data for the first and second experiment. 
- **fits/priority:** contains model predictions for the first and second experiment. 
- **helperfunctions:** functions necessary for model fitting, plots, etc. 
- **model:** all functions necessary to fit any of the models to data. 
- **tutorial:** this contains files for a lab meeting tutorial. It may not be helpful for you. 
- **plots_for_pub.m:** recreate all plots in publication
- **stats_for_pub.m** redo stats reported in publication

