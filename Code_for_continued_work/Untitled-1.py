# %%
import numpy as np  
import pandas as pd
from tqdm import tqdm

# %% [markdown]
# # Tests for distribution shifts
# ## Part 1: Non-parametric
# ### 1. Permutation tests

# %%
#Recreate the Hiring Example from R90 paper
#Set seed
#np.random.seed(1234)
#Original Data - Sample 1
#exogenous Variable U ~ N(0,1)
U = np.random.normal(0,1,500)
#sex X ~ Bernoulli(expit(U))
expit_u = (np.exp(U)/(1+np.exp(U)))
X = np.random.binomial(1, expit_u)
#age group Z ~ Bernoulli(expit(U))
Z = np.random.binomial(1, expit_u)
#education level W ~ Bernoulli(0.3)
W = np.random.binomial(1, 0.3, 500)
#hiring decision Y ~ Bernoulli(0.2*(X+Z-2*Z*X)+(1/6)*w))
Y = np.random.binomial(1, 0.2*(X+Z-2*Z*X)+(1/6)*W+0.05,500)


#Store data in a dataframe
sample_1 = pd.DataFrame({'U':U, 'X':X, 'Z':Z, 'W':W, 'Y':Y})


#np.random.seed(1234)
#New domain - Sample 2
#exogenous Variable U ~ N(0,1)
U = np.random.normal(0,1,500)
#sex X ~ Bernoulli(expit(U))
expit_u = (np.exp(U)/(1+np.exp(U)))
X = np.random.binomial(1, expit_u)
#age group Z ~ Bernoulli(expit(U))
Z = np.random.binomial(1, expit_u)
#education level W ~ Bernoulli(0.3)
W = np.random.binomial(1, 0.3, 500)
#hiring decision Y ~ Bernoulli(0.2*(X+Z-2*Z*X)+(1/6)*w))
Y = np.random.binomial(1,1,500)


#Store data in a dataframe
sample_2 = pd.DataFrame({'U':U, 'X':X, 'Z':Z, 'W':W, 'Y':Y})


# %%
def permutation_test(sample_1, sample_2, n_permutations=10000):
    # Compute the test statistic for the observed data
    group_cols = ["X", "Z", "W"]
    sample_1_grouped = sample_1.groupby(group_cols)['Y'].sum().reset_index()
    sample_2_grouped = sample_2.groupby(group_cols)['Y'].sum().reset_index()
    
    # Calculate the observed difference in probabilities P(Y=1|X,Z,W) between sample_1 and sample_2
    differences = (sample_1_grouped['Y'].values / len(sample_1)) - (sample_2_grouped['Y'].values / len(sample_2))
    
    # Initialize the results DataFrame
    results = pd.DataFrame({
        'P_a(Y=1|X,Z,W) - P_b(Y=1|X,Z,W)': differences,
        'P-Values': np.zeros(len(differences)),
        'Decision': np.zeros(len(differences))
    })
    
    # Pool the data
    pooled_data = pd.concat([sample_1, sample_2], ignore_index=True)
    
    # Calculate group indices once (since we repeatedly group by the same columns)
    pooled_grouped = pooled_data.groupby(group_cols).size().reset_index().drop(columns=0)
    
    # Precompute the sizes of the samples
    sample_1_size = len(sample_1)
    sample_2_size = len(sample_2)
    
    # Vectorized p-value calculation
    count_extreme = np.zeros(len(differences))

    for i in tqdm(range(n_permutations), desc="Permutations"):
        # Shuffle the pooled data and split into two shuffled samples
        shuffled_data = pooled_data.sample(frac=1).reset_index(drop=True)
        
        shuffled_sample_1 = shuffled_data.iloc[:sample_1_size]
        shuffled_sample_2 = shuffled_data.iloc[sample_1_size:]
        
        # Group and calculate the differences for this permutation
        shuffled_sample_1_grouped = shuffled_sample_1.groupby(group_cols)['Y'].sum().reset_index()
        shuffled_sample_2_grouped = shuffled_sample_2.groupby(group_cols)['Y'].sum().reset_index()
        
        # Compute permuted differences
        temp_difference = (shuffled_sample_1_grouped['Y'].values / sample_1_size) - \
                          (shuffled_sample_2_grouped['Y'].values / sample_2_size)
        
        # Count how often the permuted differences are more extreme than the observed
        count_extreme += (np.abs(temp_difference) > np.abs(differences))

    # Calculate p-values based on the extreme counts
    results['P-Values'] = count_extreme / n_permutations
    
    # Decision: reject if p-value < 0.05
    results['Decision'] = np.where(results['P-Values'] < 0.05, 'reject', 'no reject')

    #Add another column to the results dataframe to allow for Bonferroni-Holm correction
    results['Adjusted P-Values'] = np.zeros(len(differences))
    #Add another column to the results dataframe to indicate the Bonferroni-Holm decision
    results['Bonferroni-Holm Decision'] = np.zeros(len(differences))

    #Bonferroni-Holm correction
    #Sort the p-values and only keep the indices
    temp_sort_p_values = np.argsort(results['P-Values']).values
    j = 1
    for index in temp_sort_p_values:
        current_p_value = results['P-Values'][index]
        bonferonni_holm_factor = len(results) - j + 1
        adjusted_p_value = current_p_value * bonferonni_holm_factor
        non_significant = False
        while non_significant == False:
            if adjusted_p_value < 0.05:
                results['Adjusted P-Values'][index] = adjusted_p_value
                results['Bonferroni-Holm Decision'][index] = 'reject'
                non_significant = False
            else:
                results['Adjusted P-Values'][index:] = "terminated early"
                results['Bonferroni-Holm Decision'][index:] = 'no reject'
                non_significant = True
        j = j + 1
    
    
    return results

# %%
test = permutation_test(sample_1, sample_2, n_permutations=10000)
test

# %% [markdown]
# - seems to be sensitive to n_permutations (the more the more accurately it detects a change)
# - seems to not be able to detect small differences

# %% [markdown]
# # Non - parametric
# ## Permutation test (1) 

# %%
#Compute test statistic
def test_statistic(sample_1,sample_2):


    #Create contingency tables and compute P(Y=1|X,Z,W) for both samples 
    contingency_table_1 = sample_1.groupby(['X','Z','W'])["Y"].mean().reset_index()
    contingency_table_1.columns = ('X','Z','W','P(Y=1|X,Z,W)')
    contingency_table_2 = sample_2.groupby(['X','Z','W'])["Y"].mean().reset_index()
    contingency_table_2.columns = ('X','Z','W','P(Y=1|X,Z,W)')

    #Create empty table, which copies the (X,Z,W) columns from the first contingency table
    test_statistic_table = contingency_table_1[['X','Z','W']]
    #Now add P_a(Y=1|X,Z,W) and P_b(Y=1|X,Z,W) to the table and a third column with the difference
    test_statistic_table['P_a(Y=1|X,Z,W)'] = contingency_table_1['P(Y=1|X,Z,W)']
    test_statistic_table['P_b(Y=1|X,Z,W)'] = contingency_table_2['P(Y=1|X,Z,W)']
    test_statistic_table['Difference'] = contingency_table_1['P(Y=1|X,Z,W)'] - contingency_table_2['P(Y=1|X,Z,W)']
    
    #return the test statistic table
    return test_statistic_table


def permutation_test_scratch(sample_1,sample_2,number_of_permutations=10000):
    
    #compute test_statistic
    test_statistic_table = test_statistic(sample_1,sample_2)
    
    #Pool the data
    pooled_data = pd.concat([sample_1,sample_2])
    
    #for loop to create permutations and compute the difference in P(Y=1|X,Z,W) for each permutation
    #The permuted differences should be stores as (8x6xnumber_of_permutations) array, where the first two dimensions are the (X,Z,W) combinations, P_a(Y=1|X,Z,W), P_b(Y=1|X,Z,W), Difference and the third dimension is the number of permutations
    #Essentially we have a table for each permutation
    permuted_differences = np.zeros((8,6,number_of_permutations))
    for i in range(number_of_permutations):
        #Create two shuffled samples
        shuffled_sample_1 = pooled_data.sample(frac=0.5)
        shuffled_sample_2 = pooled_data.drop(shuffled_sample_1.index)

        #Create contingency tables and compute P(Y=1|X,Z,W) for both samples
        contingency_table_1 = shuffled_sample_1.groupby(['X','Z','W'])["Y"].mean().reset_index()
        contingency_table_1.columns = ('X','Z','W','P(Y=1|X,Z,W)')
        contingency_table_2 = shuffled_sample_2.groupby(['X','Z','W'])["Y"].mean().reset_index()
        contingency_table_2.columns = ('X','Z','W','P(Y=1|X,Z,W)')

        #Create empty table, which copies the (X,Z,W) columns from the first contingency table
        temp = contingency_table_1[['X','Z','W']]
        #Now add P_a(Y=1|X,Z,W) and P_b(Y=1|X,Z,W) to the table and a third column with the difference
        temp['P_a(Y=1|X,Z,W)'] = contingency_table_1['P(Y=1|X,Z,W)']
        temp['P_b(Y=1|X,Z,W)'] = contingency_table_2['P(Y=1|X,Z,W)']
        temp['Difference'] = contingency_table_1['P(Y=1|X,Z,W)'] - contingency_table_2['P(Y=1|X,Z,W)']

        #Store the test_statistic table in the permuted_differences array
        permuted_differences[:,:,i] = temp.values

        #Compute the p-value
        # p = Number of times the absolute permuted difference is greater than the absolute observed difference / number of permutations

        for i in range(len(test_statistic_table)):
            #fix first dimension of permuted_differences and fix the 6th column as this is where the difference is stored
            permuted_differences_fixed = permuted_differences[i,5,:]
            #compute p-value for that (X,Z,W) combination
            p_value = (np.abs(permuted_differences_fixed) > np.abs(test_statistic_table['Difference'][i])).sum()/number_of_permutations
            
            if p_value == test_statistic_table['Difference'][i]:
                rejection = "Degenerate"
            else:
                rejection = p_value < 0.05  

            #store p-value and rejection in the test_statistic_table
            test_statistic_table.loc[i,'p-value'] = p_value
            test_statistic_table.loc[i,'Reject'] = rejection

            #add another p-value column "p-value (Bonferroni)" which is the p-value adjusted for multiple testing
            if p_value*len(test_statistic_table) < 1:
                test_statistic_table.loc[i,'p-value (Bonferroni)'] = p_value*8
            else:
                test_statistic_table.loc[i,'p-value (Bonferroni)'] = 1
            #similarly add a rejection column for the Bonferroni adjusted p-value
            #first we check whether we have a degenrate case
            if rejection == "Degenerate":
                test_statistic_table.loc[i,'Reject (Bonferroni)'] = "Degenerate"
            else:
                test_statistic_table.loc[i,'Reject (Bonferroni)'] = p_value*8 < 0.05
         

                

    return test_statistic_table

# %%
#Run the permutation test
test_statistic_table = permutation_test_scratch(sample_1,sample_2,number_of_permutations=10000)
test_statistic_table
    

# %% [markdown]
# ## Equivalance testing Permutation test (2) 

# %%
def two_one_sided_permutation_test(sample1, sample2, number_permutations,margin = 0.01):

    #compute test_statistic
    test_statistic_table = test_statistic(sample1,sample2)
    
    #create empty array to store the permuted differences of dimensions (rows:len(test_statistic_table),columns:1 and number_permutations)
    #each row corresponds to a (X,Z,W) combination, and the column stores the difference in P(Y=1|X,Z,W) for each permutation.
    #The third dimension is the number of permutations
    permuted_differences = np.zeros((len(test_statistic_table),1,number_permutations))

    #pool the data
    pooled_data = pd.concat([sample1,sample2])

    #create the permutations and compute the difference in P(Y=1|X,Z,W) for each permutation
    for i in range(number_permutations):

        #Create two shuffled samples
        shuffled_sample_1 = pooled_data.sample(frac=0.5)
        shuffled_sample_2 = pooled_data.drop(shuffled_sample_1.index)

        #Compute the difference in P(Y=1|X,Z,W) for each permutation and store as a (len(test_statistic_table),1) array
        differences = shuffled_sample_1.groupby(['X','Z','W'])["Y"].mean().reset_index()["Y"].values - shuffled_sample_2.groupby(['X','Z','W'])["Y"].mean().reset_index()["Y"].values
        differences = differences.reshape((8,1))
        #Store the test_statistic table in the permuted_differences array
        permuted_differences[:,:,i] = differences

    #compute the two p-values (p_lower and p_upper) for each (X,Z,W) combination

    for i in range(len(test_statistic_table)):
        #fix first dimension of permuted_differences and fix the 6th column as this is where the difference is stored
        permuted_differences_fixed = permuted_differences[i,0,:]
        #compute p-value for that (X,Z,W) combination
        p_lower = (permuted_differences_fixed < test_statistic_table['Difference'][i] - margin).sum()/number_permutations
        p_upper = (permuted_differences_fixed > test_statistic_table['Difference'][i] + margin).sum()/number_permutations

        #store p-values in the test_statistic_table
        test_statistic_table.loc[i,'p-value (lower)'] = p_lower
        test_statistic_table.loc[i,'p-value (upper)'] = p_upper

        #compute the two rejection decisions
        reject_lower = p_lower < 0.05
        reject_upper = p_upper < 0.05

        #store the rejection decisions in the test_statistic_table
        test_statistic_table.loc[i,'Reject (lower)'] = reject_lower
        test_statistic_table.loc[i,'Reject (upper)'] = reject_upper

    return test_statistic_table





