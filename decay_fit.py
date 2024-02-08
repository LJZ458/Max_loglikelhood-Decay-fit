import numpy as np
from sympy import *
import matplotlib.pyplot as plt

f=open('20Mggate1.txt',"r")
lines=f.readlines()
raw=[]
for x in lines:
    raw.append(x.split(' '))

decay_time = []
decay_counts = []
data = []


for item in raw:
    data.append(item)

for i in data:    
        decay_time.append(int((i[0][1:-1])))
        if len(i[-1])==1:
             decay_counts.append(int(i[-1]))
        else:
            decay_counts.append(int(i[-1][0:-1]))

f.close()
# a = np.loadtxt("Total3.dat")
# decay_counts = np.transpose(a)[1]
# # decay_counts = np.transpose(a)[0:300]

# # print(decay_counts)
# # decay_counts = decay_counts[0:-1]
y_i = decay_counts

################################START TIME ----------------------END TIME################################################################
bin_size = 0.008


#linspace(start_time,endtime,number of bins)
# t_val = []
# for i in range(0,len(y_i)):
#     t_val.append(bin_size*i)
# t_val = np.array(t_val)
t_val = np.linspace(bin_size/2, len(decay_counts)*bin_size+bin_size/2,len(decay_counts),endpoint = False)

# print(t_val)
############################################################################################################################################################
################################  PARAMTERS FOR CURVE FITTING###############################################################################################


Max_Iterations = 200
tolerance = 0.000001
decay_cons_20Na = np.log(2)/0.44790
decay_cons_20Mg = np.log(2)/0.075
intensity_1_est = decay_counts[0]
####################################################iNITIAL GUESS PARAMETERS#############################################################################
#IF THE YOU WANT TO FIX THE PARAMETER JUST ENTER 0 HERE AND REPLACE THE PARAMETER AS CONSTANT YOU WANT IT TO FIXED IN THE FUNCTION BELOW, THERE MIGHT BE ISSUE IF
#YOU SPECIFIED THE PARAMETER HERE AS 0 BUT STILL INCLUDING IT IN THE FUNCTION DEFINED.
lambda_1_val_init, lambda_2_val_init, lambda_3_val_init,intensity_1_val_init,intensity_2_val_init,intensity_3_val_init,backgrounds_val_init = decay_cons_20Mg,0,0,intensity_1_est,0,0,0


# decay fit using maximum log-likelyhood approach where parameters are found using algorithm based on Levenberg-Marquardt Method

#compute chi square for maximum log likelihood:

def chi_square_ml(y_i,y_fit):
    value = 0,
    if len(y_i) != len(y_fit):
        print("different array length found for y_i and y_fit, chi square computation failed")
        return 0
    else:
        for i in range(0,len(y_i)):
            if y_i[i] == 0.:
                value += (y_fit[i])
            else:
                value += (y_fit[i]- y_i[i] + y_i[i]*np.log(y_i[i]/y_fit[i]))
    return value*2

def exp_chi_square(y_i,y_fit):
    value = 0
    for i in range(0,len(y_i)):
            if y_i[i] < 4.2:
                value = -2*y_i[i]*np.log(y_i[i]) + y_i[i]**2*np.log(4)  
                value += np.absolute(y_fit[i])
            else:
                value += np.absolute(y_fit[i]- y_i[i] + y_i[i]*np.log(y_i[i]/y_fit[i]))
    return value*2



#partial derivative computations are done using external package by sympy, specifications can be foubd at https://docs.sympy.org/latest/tutorials/index.html#tutorials
#initialization of variables 

lambda_1, lambda_2, lambda_3,t,intensity_1,intensity_2,intensity_3,backgrounds  = symbols('lambda_1 lambda_2 lambda_3 t intensity_1 intensity_2 intensity_3 backgrounds')
init_printing(use_unicode=True)


##################################################create functions using the parameters initialized#########################################################
##################################IF THE PARAMETER YOU SPECIFIED IS FIXED AT A CONSTANT, DO NOT ENETER THE PARAMETER HERE, USE THE CONSTANT VALUE DIRECTLY#############
y_fit_func =  intensity_1*exp(-lambda_1*t)+ 0.7*  (decay_cons_20Na/(decay_cons_20Na-lambda_1))*  intensity_1 *   ( exp(-lambda_1*t) -exp(-decay_cons_20Na*t))
# y_fit_func =  intensity_1*bin_size*exp(-lambda_1*t) + backgrounds*bin_size
#note by using the package, the above function can be printed but can not be used in calculations, in order to create a real function that returns value, need to use function lambdify to
#reutrn a lambda like function as usual
y_fit_func_val = lambdify([lambda_1, lambda_2, lambda_3,t,intensity_1,intensity_2,intensity_3,backgrounds],y_fit_func)
gradient = diff(y_fit_func, lambda_1)
print(gradient)
gradient_val = lambdify([lambda_1, lambda_2, lambda_3,t,intensity_1,intensity_2,intensity_3,backgrounds],gradient)


#---------------------------------------------------------GRADIENT---COMPUTATIONS----------------------------------------------------------------------------
def dy_dlambda_1(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val):
    dy_dlambda_1_value  = lambdify([lambda_1, lambda_2, lambda_3,t,intensity_1,intensity_2,intensity_3,backgrounds],diff(y_fit_func, lambda_1))
    return dy_dlambda_1_value(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)


def dy_dlambda_2(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val):
    dy_dlambda_2_value  = lambdify([lambda_1, lambda_2, lambda_3,t,intensity_1,intensity_2,intensity_3,backgrounds],diff(y_fit_func, lambda_2))
    return dy_dlambda_2_value(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)


def dy_dlambda_3(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val):
    dy_dlambda_3_value  = lambdify([lambda_1, lambda_2, lambda_3,t,intensity_1,intensity_2,intensity_3,backgrounds],diff(y_fit_func, lambda_3))
    return dy_dlambda_3_value(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)


def dy_di_1(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val):
    dy_di_1_value  = lambdify([lambda_1, lambda_2, lambda_3,t,intensity_1,intensity_2,intensity_3,backgrounds],diff(y_fit_func, intensity_1))
    return dy_di_1_value(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)


def dy_di_2(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val):
    dy_di_2_value  = lambdify([lambda_1, lambda_2, lambda_3,t,intensity_1,intensity_2,intensity_3,backgrounds],diff(y_fit_func, intensity_2))
    return dy_di_2_value(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)


def dy_di_3(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val):
    dy_di_3_value  = lambdify([lambda_1, lambda_2, lambda_3,t,intensity_1,intensity_2,intensity_3,backgrounds],diff(y_fit_func, intensity_3))
    return dy_di_3_value(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)


def dy_dbackgrounds(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val):
    dy_dbackgrounds_value  = lambdify([lambda_1, lambda_2, lambda_3,t,intensity_1,intensity_2,intensity_3,backgrounds],diff(y_fit_func, backgrounds))
    return dy_dbackgrounds_value(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------RETURNS--FITTED--VALUE--WITH--CURRENT--FIT--PARAMETERS--------------------------------------------------------------------------------
def fit_function_val(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val):
    y_fit = []
    for i in range(0,len(t_val)):
        y_fit.append( y_fit_func_val(lambda_1_val, lambda_2_val, lambda_3_val,t_val[i],intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val))
    return y_fit


def primary_fit_function_val(lambda_1_val,t_val,intensity_1_val):
    y_fit = []
    for i in range(0,len(t_val)):
        y_fit.append(intensity_1_val*lambda_1_val*exp(-lambda_1_val*t_val[i]))
    return y_fit


def secondary_fit_function_val(lambda_1_val, lambda_2_val,t_val,intensity_1_val,intensity_2_val):
    y_fit = []
    for i in range(0,len(t_val)):
        y_fit.append(0.7 *lambda_1_val*(lambda_2_val/(lambda_2_val-lambda_1_val))*   intensity_1_val   *   (exp(-lambda_1_val*t_val[i]) - exp(-lambda_2_val*t_val[i])) )
    return y_fit


def fit_backgrounds(backgrounds_val,t_val):
    fit = []
    for i in range(0,len(t_val)):
        fit.append(backgrounds_val)
    return fit




def log_fit_function_val(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val):
    y_fit = []
    for i in range(0,len(t_val)):
        y_fit.append(np.log(float( y_fit_func_val(lambda_1_val, lambda_2_val, lambda_3_val,t_val[i],intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val))))
    return y_fit





def log_primary_fit_function_val(lambda_1_val,t_val,intensity_1_val):
    y_fit = []
    for i in range(0,len(t_val)):
        y_fit.append(np.log(float(intensity_1_val*lambda_1_val*exp(-lambda_1_val*t_val[i]))))
    return y_fit


def log_secondary_fit_function_val(lambda_1_val, lambda_2_val,t_val,intensity_1_val,intensity_2_val):
    y_fit = []
    for i in range(0,len(t_val)):
        y_fit.append(np.log(float(0.7*lambda_1_val *(lambda_2_val/(lambda_2_val-lambda_1_val))*   intensity_1_val   *   (exp(-lambda_1_val*t_val[i]) - exp(-lambda_2_val*t_val[i])) )))
    return y_fit


#---------------------------------DEFINE--FUNCTION--OF--EXTREMUM--VECTOR----------------------------------------------------------------------------------------------



def extrm_vector(y_i,lambda_1_val,lambda_2_val,lambda_3_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val,t_val):
    y_fit = fit_function_val(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
    vector = []
    dy_dlambda_1_val = dy_dlambda_1(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
    dy_dlambda_2_val = dy_dlambda_2(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
    dy_dlambda_3_val = dy_dlambda_3(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
    dy_di_1_val =  dy_di_1(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
    dy_di_2_val = dy_di_2(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
    dy_di_3_val = dy_di_3(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
    dy_dbackgrounds_val =  dy_dbackgrounds(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
    # dy_dbackgrounds_val =  dy_dbackgrounds(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)

    # print(type(dy_dlambda_1_val))
    if lambda_2_val_init==0 or  type(dy_dlambda_2_val) != np.ndarray :
        dy_dlambda_2_val = np.zeros(len(t_val))
    if intensity_2_val_init == 0 or  type(dy_di_2_val) != np.ndarray:
        dy_di_2_val = np.zeros(len(t_val))
    if lambda_3_val_init==0 or  type(dy_dlambda_3_val) != np.ndarray:
        dy_dlambda_3_val = np.zeros(len(t_val))
    if intensity_3_val_init == 0 or type(dy_di_3_val) != np.ndarray:
        dy_di_3_val  = np.zeros(len(t_val))

    if backgrounds_val_init == 0:
        dy_dbackgrounds_val =  np.zeros(len(t_val))
    elif backgrounds_val_init != 0 and type(dy_dbackgrounds) != np.ndarray:
        dy_dbackgrounds_val =  np.ones(len(t_val))*dy_dbackgrounds(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)

    
    val = 0
    for i in range(0,len(y_i)):
        val += (y_i[i] - y_fit[i])/y_fit[i] * dy_dlambda_1_val[i]
    vector.append(val)
    val = 0

    for i in range(0,len(y_i)):
        val += (y_i[i] - y_fit[i])/y_fit[i] * dy_dlambda_2_val[i]
    vector.append(val)
    val = 0

    for i in range(0,len(y_i)):
        val += (y_i[i] - y_fit[i])/y_fit[i] * dy_dlambda_3_val[i]
    vector.append(val)
    val = 0

    for i in range(0,len(y_i)):
        val += (y_i[i] - y_fit[i])/y_fit[i] * dy_di_1_val[i]
    vector.append(val)
    val = 0

    for i in range(0,len(y_i)):
        val += (y_i[i] - y_fit[i])/y_fit[i] * dy_di_2_val[i]
    vector.append(val)
    val = 0

    for i in range(0,len(y_i)):
        val += (y_i[i] - y_fit[i])/y_fit[i] * dy_di_3_val[i]
    vector.append(val)
    val = 0

    for i in range(0,len(y_i)):
        val += (y_i[i] - y_fit[i])/y_fit[i] * dy_dbackgrounds_val[i]
    vector.append(val)
    # print(vector)

    test = []
    test.append(lambda_1_val_init)
    test.append(lambda_2_val_init)
    test.append(lambda_3_val_init)
    test.append(intensity_1_val_init)
    test.append(intensity_2_val_init)
    test.append(intensity_3_val_init)
    test.append(backgrounds_val_init)

    
    new_vector = []
    for i in range(0,len(test)):
        if test[i] != 0.:
            new_vector.append(vector[i])

    return new_vector




#VECTOR COMPONENT = [beta_lambda_1,beta_lambda_2,beta_lambda_3,beta_intesity_1,beta_intensity_2,beta_intensity_3,beta_backgrounds]


#----------------------------------------------------------CURVATURE--MATRIX--COMPUTATION-------------------------------------------------------------

def curvature_matrix(lambda_1_val,lambda_2_val,lambda_3_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val,t_val):
    y_i = fit_function_val(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
    matrix = []

    val_1 = 0
    val_2 = 0
    val_3 = 0
    val_4 = 0
    val_5 = 0
    val_6 = 0
    val_7 = 0
    dy_dlambda_1_val = dy_dlambda_1(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
    dy_di_1_val = dy_di_1(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
    
    if lambda_2_val_init==0:
        dy_dlambda_2_val = np.zeros(len(t_val))
    else:
        dy_dlambda_2_val = dy_dlambda_2(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
        if type(dy_dlambda_2_val) != np.ndarray:
            dy_dlambda_2_val = np.zeros(len(t_val))

    if intensity_2_val_init == 0:
        dy_di_2_val = np.zeros(len(t_val))
    else:
        dy_di_2_val = dy_di_2(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
        if type(dy_di_2_val) != np.ndarray :
            dy_di_2_val = np.zeros(len(t_val))
    if lambda_3_val_init==0 :
        dy_dlambda_3_val = np.zeros(len(t_val))
    else:
        dy_dlambda_3_val = dy_dlambda_3(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
        if type(dy_dlambda_3_val) != np.ndarray:
            dy_dlambda_3_val = np.zeros(len(t_val))
    if intensity_3_val_init == 0:
        dy_di_3_val = np.zeros(len(t_val))
    else:
        dy_di_3_val = dy_di_3(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)
        if type(dy_di_3_val) != np.ndarray:
            dy_di_3_val = np.zeros(len(t_val))
    dy_dbackgrounds_val =  dy_dbackgrounds(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)

    if backgrounds_val_init==0:
        dy_dbackgrounds_val =  np.zeros(len(t_val))
    elif backgrounds_val_init!=0 and type(dy_dbackgrounds) != np.ndarray:
        dy_dbackgrounds_val =  np.ones(len(t_val))*dy_dbackgrounds(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val)

#-----------------------------------------------row-1-compute--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
    row_1 = []

    for i in range(0,len(y_i)):
        val_1 += 1/y_i[i]*dy_dlambda_1_val[i]*dy_dlambda_1_val[i]
        val_2 += 1/y_i[i]*dy_dlambda_1_val[i]*dy_dlambda_2_val[i]
        val_3 += 1/y_i[i]*dy_dlambda_1_val[i]*dy_dlambda_3_val[i]
        val_4 += 1/y_i[i]*dy_dlambda_1_val[i]*dy_di_1_val[i]
        val_5 += 1/y_i[i]*dy_dlambda_1_val[i]*dy_di_2_val[i]
        val_6 += 1/y_i[i]*dy_dlambda_1_val[i]*dy_di_3_val[i]
        val_7 += 1/y_i[i]*dy_dlambda_1_val[i]*dy_dbackgrounds_val[i]
    
    row_1.append(val_1)
    row_1.append(val_2)
    row_1.append(val_3)
    row_1.append(val_4)
    row_1.append(val_5)
    row_1.append(val_6)
    row_1.append(val_7)


#-----------------------------------------row-2-compute------------------------------------------------------------------------------------------------
    val_1 = 0
    val_2 = 0
    val_3 = 0
    val_4 = 0
    val_5 = 0
    val_6 = 0
    val_7 = 0
    row_2 = []
    for i in range(0,len(y_i)):
        val_1 += 1/y_i[i]*dy_dlambda_2_val[i]*dy_dlambda_1_val[i]
        val_2 += 1/y_i[i]*dy_dlambda_2_val[i]*dy_dlambda_2_val[i]
        val_3 += 1/y_i[i]*dy_dlambda_2_val[i]*dy_dlambda_3_val[i]
        val_4 += 1/y_i[i]*dy_dlambda_2_val[i]*dy_di_1_val[i]
        val_5 += 1/y_i[i]*dy_dlambda_2_val[i]*dy_di_2_val[i]
        val_6 += 1/y_i[i]*dy_dlambda_2_val[i]*dy_di_3_val[i]
        val_7 += 1/y_i[i]*dy_dlambda_2_val[i]*dy_dbackgrounds_val[i]
    row_2.append(val_1)
    row_2.append(val_2)
    row_2.append(val_3)
    row_2.append(val_4)
    row_2.append(val_5)
    row_2.append(val_6)
    row_2.append(val_7)


#-----------------------------------------row-3-compute------------------------------------------------------------------------------------------------
    val_1 = 0
    val_2 = 0
    val_3 = 0
    val_4 = 0
    val_5 = 0
    val_6 = 0
    val_7 = 0
    row_3 = []
    for i in range(0,len(y_i)):
        val_1 += 1/y_i[i]*dy_dlambda_3_val[i]*dy_dlambda_1_val[i]
        val_2 += 1/y_i[i]*dy_dlambda_3_val[i]*dy_dlambda_2_val[i]
        val_3 += 1/y_i[i]*dy_dlambda_3_val[i]*dy_dlambda_3_val[i]
        val_4 += 1/y_i[i]*dy_dlambda_3_val[i]*dy_di_1_val[i]
        val_5 += 1/y_i[i]*dy_dlambda_3_val[i]*dy_di_2_val[i]
        val_6 += 1/y_i[i]*dy_dlambda_3_val[i]*dy_di_3_val[i]
        val_7 += 1/y_i[i]*dy_dlambda_3_val[i]*dy_dbackgrounds_val[i]
    row_3.append(val_1)
    row_3.append(val_2)
    row_3.append(val_3)
    row_3.append(val_4)
    row_3.append(val_5)
    row_3.append(val_6)
    row_3.append(val_7)

#-----------------------------------------row-4-compute------------------------------------------------------------------------------------------------
    val_1 = 0
    val_2 = 0
    val_3 = 0
    val_4 = 0
    val_5 = 0
    val_6 = 0
    val_7 = 0
    row_4 = []
    for i in range(0,len(y_i)):
        val_1 += 1/y_i[i]*dy_di_1_val[i]*dy_dlambda_1_val[i]
        val_2 += 1/y_i[i]*dy_di_1_val[i]*dy_dlambda_2_val[i]
        val_3 += 1/y_i[i]*dy_di_1_val[i]*dy_dlambda_3_val[i]
        val_4 += 1/y_i[i]*dy_di_1_val[i]*dy_di_1_val[i]
        val_5 += 1/y_i[i]*dy_di_1_val[i]*dy_di_2_val[i]
        val_6 += 1/y_i[i]*dy_di_1_val[i]*dy_di_3_val[i]
        val_7 += 1/y_i[i]*dy_di_1_val[i]*dy_dbackgrounds_val[i]
    row_4.append(val_1)
    row_4.append(val_2)
    row_4.append(val_3)
    row_4.append(val_4)
    row_4.append(val_5)
    row_4.append(val_6)
    row_4.append(val_7)

#-----------------------------------------row-4-compute------------------------------------------------------------------------------------------------
    val_1 = 0
    val_2 = 0
    val_3 = 0
    val_4 = 0
    val_5 = 0
    val_6 = 0
    val_7 = 0
    row_5 = []
    for i in range(0,len(y_i)):
        val_1 += 1/y_i[i]*dy_di_2_val[i]*dy_dlambda_1_val[i]
        val_2 += 1/y_i[i]*dy_di_2_val[i]*dy_dlambda_2_val[i]
        val_3 += 1/y_i[i]*dy_di_2_val[i]*dy_dlambda_3_val[i]
        val_4 += 1/y_i[i]*dy_di_2_val[i]*dy_di_1_val[i]
        val_5 += 1/y_i[i]*dy_di_2_val[i]*dy_di_2_val[i]
        val_6 += 1/y_i[i]*dy_di_2_val[i]*dy_di_3_val[i]
        val_7 += 1/y_i[i]*dy_di_2_val[i]*dy_dbackgrounds_val[i]
    row_5.append(val_1)
    row_5.append(val_2)
    row_5.append(val_3)
    row_5.append(val_4)
    row_5.append(val_5)
    row_5.append(val_6)
    row_5.append(val_7)

#-----------------------------------------row-5-compute------------------------------------------------------------------------------------------------
    val_1 = 0
    val_2 = 0
    val_3 = 0
    val_4 = 0
    val_5 = 0
    val_6 = 0
    val_7 = 0
    row_6 = []
    for i in range(0,len(y_i)):
        val_1 += 1/y_i[i]*dy_di_3_val[i]*dy_dlambda_1_val[i]
        val_2 += 1/y_i[i]*dy_di_3_val[i]*dy_dlambda_2_val[i]
        val_3 += 1/y_i[i]*dy_di_3_val[i]*dy_dlambda_3_val[i]
        val_4 += 1/y_i[i]*dy_di_3_val[i]*dy_di_1_val[i]
        val_5 += 1/y_i[i]*dy_di_3_val[i]*dy_di_2_val[i]
        val_6 += 1/y_i[i]*dy_di_3_val[i]*dy_di_3_val[i]
        val_7 += 1/y_i[i]*dy_di_3_val[i]*dy_dbackgrounds_val[i]
    row_6.append(val_1)
    row_6.append(val_2)
    row_6.append(val_3)
    row_6.append(val_4)
    row_6.append(val_5)
    row_6.append(val_6)
    row_6.append(val_7)


#-----------------------------------------row-4-compute------------------------------------------------------------------------------------------------
    val_1 = 0
    val_2 = 0
    val_3 = 0
    val_4 = 0
    val_5 = 0
    val_6 = 0
    val_7 = 0
    row_7 = []
    for i in range(0,len(y_i)):
        val_1 += 1/y_i[i]*dy_dbackgrounds_val[i]*dy_dlambda_1_val[i]
        val_2 += 1/y_i[i]*dy_dbackgrounds_val[i]*dy_dlambda_2_val[i]
        val_3 += 1/y_i[i]*dy_dbackgrounds_val[i]*dy_dlambda_3_val[i]
        val_4 += 1/y_i[i]*dy_dbackgrounds_val[i]*dy_di_1_val[i]
        val_5 += 1/y_i[i]*dy_dbackgrounds_val[i]*dy_di_2_val[i]
        val_6 += 1/y_i[i]*dy_dbackgrounds_val[i]*dy_di_3_val[i]
        val_7 += 1/y_i[i]*dy_dbackgrounds_val[i]*dy_dbackgrounds_val[i]
    row_7.append(val_1)
    row_7.append(val_2)
    row_7.append(val_3)
    row_7.append(val_4)
    row_7.append(val_5)
    row_7.append(val_6)
    row_7.append(val_7)

    matrix.append(row_1)
    matrix.append(row_2)
    matrix.append(row_3)
    matrix.append(row_4)
    matrix.append(row_5)
    matrix.append(row_6)
    matrix.append(row_7)

    test = []
    test.append(lambda_1_val_init)
    test.append(lambda_2_val_init)
    test.append(lambda_3_val_init)
    test.append(intensity_1_val_init)
    test.append(intensity_2_val_init)
    test.append(intensity_3_val_init)
    test.append(backgrounds_val_init)

    positions = []

    for i in range(0,len(test)):
        if test[i] == 0:
            positions.append(i)
            # print(i)

    if len(positions) != 0:
        new_mat = np.delete(np.array(matrix), positions, 0)
        matrix_final = np.delete(np.array(new_mat), positions, 1)
        return matrix_final
    else:
        return matrix

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# print(fit_function_val(lambda_1_val, lambda_2_val, lambda_3_val,t_val,intensity_1_val,intensity_2_val,intensity_3_val,backgrounds_val))



#-----------------------------------------------DEFINE--FUNCTION--USING--LEVENBERG-MARQUADT-METHOD-----------------------------------------------------------------------

def curvature_prime(curvature_matrix,lambda_val):
    curvature_matrix_prime = curvature_matrix
    for i in range(0,len(curvature_matrix)):
        for j in range(0,len(curvature_matrix[0])):
            if i == j:
                curvature_matrix_prime[i][j] *= (1+lambda_val)
    return curvature_matrix_prime


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------DEFINING--STEPPEING--VECTOR------------------------------------------------------------------------
#IF WE STILL INCLUDE THOSE ZERO ENTRY ROWS AND COLOMN WE MAY ENCOUNTER DIVISON ZERO ISSUES WHEN CALCULATING MATRIX INVERSION, TO AVOID THIS ISSUE CAREFULLY REMOVES
#THOSE ROWS AND COLOUMS OF ZERO ENTRYS AND ADD THEM BACK IN THE END



def STEPPING_VEC(curvature_matrix,extremum_vector):
    step_vec = np.matmul(np.linalg.inv(curvature_matrix),np.array(extremum_vector))
    print(step_vec)
    return step_vec
#save the copy of parameter initialized by creating new set of parameters for iterations
lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr = lambda_1_val_init,lambda_2_val_init,lambda_3_val_init,intensity_1_val_init,intensity_2_val_init,intensity_3_val_init,backgrounds_val_init



param_list_init = [lambda_1_val_init,lambda_2_val_init,lambda_3_val_init,intensity_1_val_init,intensity_2_val_init,intensity_3_val_init,backgrounds_val_init]
param_list_itr = [lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr]


if lambda_2_val_init==0 :
    print("lambda2 value has been disabled")

if intensity_2_val_init == 0.:
    print("intensity 2 value has been disabled")
if lambda_3_val_init==0:
    print("lambda_3 value has been disabled")

if intensity_3_val_init == 0:
    print("intensity 3 hase been disabled")
if backgrounds_val_init == 0:
    print("backgrounds value has been disabled")


lambda_val = 0.0001


for i in range(0,Max_Iterations):
    print("half life for primary beam is: ", np.log(2)/lambda_1_val_itr  )
    backup = [lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr]
    chi2 = chi_square_ml(y_i,fit_function_val(lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,t_val,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr))
    print("iteration",i," Chi square is:", chi2[0]/(len(t_val)-(np.count_nonzero(param_list_init)  )))
    curvature = curvature_matrix(lambda_1_val_itr,lambda_2_val_itr,lambda_3_val_itr,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr,t_val)
    curvature_new = curvature_prime(curvature,lambda_val)
    etxrm = extrm_vector(y_i,lambda_1_val_itr,lambda_2_val_itr,lambda_3_val_itr,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr,t_val) 
    step_vec = STEPPING_VEC(curvature_new,etxrm)
    step_vec_new = [0,0,0,0,0,0,0]
    j=0
    for i in range(0,len(param_list_init)):
        if param_list_init[i] == 0:
            j+=1
        else:
            step_vec_new[i] = step_vec[i-j]
    # print(step_vec_new)
    param_list_backup = param_list_itr
    param_list_itr = np.add(np.array(step_vec_new),np.array(param_list_itr))
    param_diff = param_list_itr-param_list_backup
    chi2_new = chi_square_ml(y_i,fit_function_val(param_list_itr[0], param_list_itr[1], param_list_itr[2],t_val,param_list_itr[3],param_list_itr[4],param_list_itr[5],param_list_itr[6]))

    if chi2_new[0] < chi2[0]:
        if np.sum(np.absolute(param_diff)) <= tolerance:
            param_list_itr = param_list_backup
            print("chi_new smaller than chi_2, difference in parameters smaller than tolerance, iteration stopped")
            lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr = backup[0],backup[1],backup[2],backup[3],backup[4],backup[5],backup[6]
            break
        lambda_val /= 10
        lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr= param_list_itr[0],param_list_itr[1],param_list_itr[2],param_list_itr[3],param_list_itr[4],param_list_itr[5],param_list_itr[6]

    else:
        lambda_val *= 10
        
        if np.sum(np.absolute(param_diff)) <= tolerance:
                print("chi_new greater than chi_2, difference in parameters smaller than tolerance, iteration stopped")
                lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr = backup[0],backup[1],backup[2],backup[3],backup[4],backup[5],backup[6]
                param_list_itr = param_list_backup
                break
        lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr = backup[0],backup[1],backup[2],backup[3],backup[4],backup[5],backup[6]


print(param_list_itr)
curvature = curvature_matrix(lambda_1_val_itr,lambda_2_val_itr,lambda_3_val_itr,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr,t_val)
covariance = np.linalg.inv(curvature)
print("half life for primary beam is: ", np.log(2)/lambda_1_val_itr , " +-", np.log(2)/(lambda_1_val_itr**2)*np.sqrt(covariance[0][0]),"s" )

if lambda_2_val_init!=0:
    print("half life for secondary beam is " ,np.log(2)/lambda_2_val_itr , " +-", np.log(2)/(lambda_2_val_itr**2)*np.sqrt(covariance[1][1]) ,"s")

print("reduced chi_square for this fit is:",  chi2[0]/(len(t_val)-(np.count_nonzero(param_list_itr)  )))

######################################CREATE PLOTS FOR ACTIVITY CURVES###########################################################


fig = plt.figure(figsize = (6,4))
ax1 = fig.add_axes([0.05,0.2,0.90,0.75])
ax2 = fig.add_axes([0.05,0.05,0.90,0.1])


###########################################################PLOT FITTED SUMS###########################################################################################
ax1.plot(t_val,fit_function_val(lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,t_val,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr)
,color = "black", label = "fitted sum")
####################################################################PLOT_FITTED PRIMARY BEAM#########################################################################

ax1.plot(t_val,primary_fit_function_val(lambda_1_val_itr,t_val,intensity_1_val_itr), color = "darkviolet", label = "Primary Beam"    )


################################################################PLOT FITTED SECONDARY BEAM######################################################################

ax1.plot(t_val , secondary_fit_function_val(lambda_1_val_itr, decay_cons_20Na,t_val,intensity_1_val_itr,intensity_2_val_itr)   ,color = "blue", label = "Secondary Beam"     )

#############################################################PLOT BACKGROUND FIT##################################################################################

ax1.plot(t_val,fit_backgrounds(backgrounds_val_itr,t_val),color = "brown", label ="Backgrounds Fit"   )

ax1.set_title("Decay Activity Fit")

ax1.set_xlabel("time(s)")
ax1.set_ylabel("Activity")

################################################################EXPERIMENTAL DATA SCATTERD################################################################################
ax1.step(t_val,decay_counts,color = "red",where="mid",linewidth = 1.5)
ax1.set_xticks(np.arange(0,2,0.16))

ax1.legend()
ax2.scatter(t_val,(np.array(decay_counts)-np.array(fit_function_val(lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,t_val,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr)))/np.sqrt(np.array(decay_counts)))

ax2.set_ylim(-3.5,3.5)
ax2.set_yticks(np.arange(-3,4,1.))
ax2.set_xticks(np.arange(0,2,0.16))

ax2.set_ylabel("Residuals/Sigma")

#################################################################Semi-Log plots with a new figure generated##############################################################



fig2 = plt.figure(figsize = (6,4))
ax3 = fig2.add_axes([0.05,0.2,0.90,0.75])

ax3.plot(t_val,log_fit_function_val(lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,t_val,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr)
,color = "blue", label = "fitted sum")

ax3.scatter(t_val,np.log(decay_counts),color = "red")
ax3.plot(t_val,log_primary_fit_function_val(lambda_1_val_itr,t_val,intensity_1_val_itr), color = "darkviolet", label = "Primary Beam"    )
ax3.plot(t_val,log_secondary_fit_function_val(lambda_1_val_itr, decay_cons_20Na,t_val,intensity_1_val_itr,intensity_2_val_itr), color = "black", label = "Secondary Beam"    )
ax3.set_xlabel("time(s)")
ax3.set_ylabel("log Activity")
ax3.set_title("Semi-log Decay Activity Fit")
ax3.legend()

# print(fit_function_val(lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,t_val,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr))



# print( "the backgrounds is:",np.ones(len(t_val))*dy_dbackgrounds(lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,t_val,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr)
# )
# print(dy_dlambda_1(lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,t_val,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr)
# )
# print(dy_dlambda_2(lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,t_val,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr)
# )


# print(gradient_val(lambda_1_val_itr, lambda_2_val_itr, lambda_3_val_itr,t_val,intensity_1_val_itr,intensity_2_val_itr,intensity_3_val_itr,backgrounds_val_itr)
# )
# print(curvature)

# plt.show()
