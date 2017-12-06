import numpy as np 
import operator

#STEP 1: Gathering relative importance of criteria as well as alternatives by methods described in the paper.
#There were four methods in subsection 3:
# 5 point likert scale
# converting price in standard scale into trapezoidal fuzzy number 
# presence and absence of a feature
# ranking converted into   aritmetic progression 


def convert_likert_scale(likert_scale_list):
	n= len(likert_scale_list)
	rel_imp_array= np.zeros(n)

	for i in range(0, n):
		if (likert_scale_list[i]=="very important"):
			rel_imp_array[i]=1
		if (likert_scale_list[i]=="important"):
			rel_imp_array[i]=0.75
		if (likert_scale_list[i]=="neutral"):
			rel_imp_array[i]=0.5
		if (likert_scale_list[i]=="not important"):
			rel_imp_array[i]=0.25
		if (likert_scale_list[i]=="not very important"):
			rel_imp_array[i]=0

	return rel_imp_array


# a triangular fuzzy number can be defined using 4 values which I define here as:
# min_p- lower value
# optimum_lower= lower limit of the optimum range (where membership rate is 1)
# optimum_highest= upper limit of the optimum range
# max_p- upper value

def standard_scale_into_fuzzy_number(price_list, min_p, optimum_lower, optimum_higher, max_p):
	n= len(price_list)
	rel_imp_array= np.zeros(n)


	#calculating membership function for the fuzzy number from the formula
	for i in range(0, n):
		if (price_list[i]<min_p or price_list[i]>max_p):
			rel_imp_array[i]=0
		if (price_list[i]>=min_p and price_list[i]<=optimum_lower):
			rel_imp_array[i]=float(price_list[i]-min_p)/float(optimum_lower-min_p)
		if (price_list[i]> optimum_lower and price_list[i]<= optimum_higher):
			rel_imp_array[i]= 1
		if (price_list[i]>optimum_higher and price_list[i]<=max_p):
			rel_imp_array[i]= float(max_p-price_list[i])/float(max_p- optimum_higher)

	return rel_imp_array


# if a feature is present then 1, else 0
def presence_absence_feature(feature_list):

	n = len(feature_list)

	rel_imp_array = np.zeros(n)

	for i in range(0, n):
		if (feature_list[i]=="Y"):
			rel_imp_array[i]=1;

	return rel_imp_array

def rank_based_conversion(rank_list):

	n= len(rank_list)
	index_max, max_rank = max(enumerate(rank_list), key=operator.itemgetter(1))
	index_min, min_rank = min(enumerate(rank_list), key=operator.itemgetter(1))
	d= float(1)/float(max_rank-min_rank)

	rel_imp_array= np.zeros(n)

	for i in range(0,n):
		rel_imp_array[i]= float(max_rank- rank_list[i])*d

	return rel_imp_array

#STEP 2: Access relative importance of different criteria or alternative by pair wise comparison
#that is calculation of a ratio matrix
#normalise every matrix

def ratio_matrix_fun(rel_imp_array):

	n= len(rel_imp_array)

	ratio_matrix= np.zeros((n,n))

	for i in range(0,n):
		for j in range(0, n):

			if(i==j): # to avoid division of zero/zero
				ratio_matrix[i][j]=1
			else:
				ratio_matrix[i][j]= rel_imp_array[i]/rel_imp_array[j]

	ratio_matrix[ratio_matrix>= 1E308]= 1000 #to remove infinities from the matrix if any 

	return ratio_matrix

def normalise(ratio_matrix):

	ratio_matrix= ratio_matrix/np.sum(ratio_matrix)
	return ratio_matrix

# STEP 3: Calculate eigen vectors from the ratio matrix

def get_principal_eigen_vector(ratio_matrix):
    n = ratio_matrix.shape[0]
    e_vals, e_vecs = np.linalg.eig(ratio_matrix)
    max_lambda = max(e_vals)
    w = e_vecs[:, (e_vals==max_lambda)]
    w = w / np.sum(w) # Normalization

	# STEP 4: Consistency Checking
    ri = [x+1 for x in np.random.rand(12)] #initialising random ri 
    ci = (max_lambda - n) / (n - 1) # formula for ci from the paper https://www.whitman.edu/Documents/Academics/Mathematics/Klutho.pdf
    cr = ci / ri[n] 
    print('CR = %f'%cr)

    if cr >= 0.1:
        print("Failed in Consistency check.")
    return w

###############################################################
###############################################################
####################Example####################################
# GOAL: to buy the most optimum mobile phone
# 5 criteria Brand, Price, Hardware, Basic built in function, Extended Built in function
# 3 alternative
#################################################################
#################################################################
#################################################################

#in the given order ["brand", "price", "hardware", "basic_built in function", "extended built in function"]
criterion_matrix= ["very important","neutral", "important", "neutral", "not important"]

#for alternatives 1, 2, 3 respectively

Brand= [1, 2, 3]
Price= [14000, 8000, 20000]
min_p= 5000
max_p=25000
optimum_lower=10000
optimum_higher=15000
Hardware= [1, 3, 2]
Basic_functions= [3, 2, 1]
Extended_functions = [2, 3, 1]

criterion_rel_imp= convert_likert_scale(criterion_matrix)
criterion_ratio_matrix= ratio_matrix_fun(criterion_rel_imp)
criterion_ratio_matrix= normalise(criterion_ratio_matrix)
criterion_w= get_principal_eigen_vector(criterion_ratio_matrix) #5x1 matrix


brand_rel_imp = rank_based_conversion(Brand)
brand_ratio_matrix= ratio_matrix_fun(brand_rel_imp)
brand_ratio_matrix= normalise(brand_ratio_matrix)
brand_w= get_principal_eigen_vector(brand_ratio_matrix) #3x1 matrix

Price_rel_imp = standard_scale_into_fuzzy_number(Price, min_p, optimum_lower, optimum_higher, max_p)
Price_ratio_matrix= ratio_matrix_fun(Price_rel_imp)
Price_ratio_matrix= normalise(Price_ratio_matrix)
Price_w= get_principal_eigen_vector(Price_ratio_matrix) #3x1 matrix

Hardware_rel_imp = rank_based_conversion(Hardware)
Hardware_ratio_matrix= ratio_matrix_fun(Hardware_rel_imp)
Hardware_ratio_matrix= normalise(Hardware_ratio_matrix)
Hardware_w= get_principal_eigen_vector(Hardware_ratio_matrix) #3x1 matrix

Basic_functions_rel_imp = rank_based_conversion(Basic_functions)
Basic_functions_ratio_matrix= ratio_matrix_fun(Basic_functions_rel_imp)
Basic_functions_ratio_matrix= normalise(Basic_functions_ratio_matrix)
Basic_functions_w= get_principal_eigen_vector(Basic_functions_ratio_matrix) #3x1 matrix

Extended_functions_rel_imp = rank_based_conversion(Extended_functions)
Extended_functions_ratio_matrix= ratio_matrix_fun(Extended_functions_rel_imp)
Extended_functions_ratio_matrix= normalise(Extended_functions_ratio_matrix)
Extended_functions_w= get_principal_eigen_vector(Extended_functions_ratio_matrix) #3x1 matrix

alternative_matrix= np.concatenate([ brand_w, Price_w , Hardware_w, Basic_functions_w,Extended_functions_w,], axis=1) 
# 3x5 matrix

scores= np.dot(alternative_matrix,criterion_w) #(3x5 * 5x1)

print scores
