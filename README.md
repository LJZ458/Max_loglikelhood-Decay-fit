


decay_fit.py is a code used for fitting radioactive decay based on maximum-log likelyhood written in python. 
The package required for running this code is :
1.numpy
2.sympy
3.matplotlib.pyplot

This code works for any activity function and requires a manual input of equation in line 82 called y_fit_function.

There are up to 7 parameters that can be used, parameters names are given as:


lambda_1, lambda_2, lambda_3,t,intensity_1,intensity_2,intensity_3,backgrounds

However, these parameteres can be anything, do not get restricted by thir name. lambda_3 could also be intensity. The only requirement is to eneter the correct parameter name in the equation.




 
Initialization parameters:

All 7 initial guesses has to be given at line 55 with correct order.
If you want to fix the parameter, enter 0 for the initial guess and replace the parameter in the function with the constant you want it to be fixed.

If you choose the initial guess to be 0 for some parameter you are fitting and it present in the fit function,there will be an issue with the fit as the code get confused whether you are fixing the parameter or not.



You have to manually input the function for primary beam fit ,secondary beam fit and so on from line 129 - 155.





 
