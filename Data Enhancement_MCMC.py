# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 18:54:40 2024

@author: LinJiamin
"""

import numpy as np
import pymc as pm
import pytensor.tensor as at
import arviz as az
import pandas as pd

def trace_to_dataframe(trace):
    """
    Converts PyMC trace to a pandas DataFrame.
    """
    data = az.from_pymc(trace)
    return az.extract(data).to_dataframe()

if __name__ == "__main__":

    df = pd.read_excel('Data/MCMC_INPUT_Lin2025.xlsx')

    
    melt_components = ['SIO2_melt', 'TIO2_melt', 'AL2O3_melt', 'FEOT_melt',
                       'CAO_melt', 'MGO_melt', 'NA2O_melt', 'K2O_melt']
    plag_components = ['SiO2_plag', 'TiO2_plag', 'Al2O3_plag', 'FeOt_plag',
                       'MgO_plag', 'CaO_plag', 'Na2O_plag', 'K2O_plag']

    melt_sd_components = [comp + '_SD' for comp in melt_components]
    plag_sd_components = [comp + '_SD' for comp in plag_components]

    other_columns = ['SUM_melt', 'SUM_plag', 'T', 'P', 'H2O']

    columns_needed = ['SAMPLE NAME'] + melt_components + melt_sd_components + \
                     plag_components + plag_sd_components + other_columns

    data = df[columns_needed]

    
    for col in melt_components + plag_components + melt_sd_components + plag_sd_components + other_columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    
    for comp, sd_comp in zip(melt_components + plag_components, melt_sd_components + plag_sd_components):
        condition = (data[comp] != 0) & (data[sd_comp] == 0)
        data.loc[condition, sd_comp] = data.loc[condition, comp] * 0.02

    
    for comp, sd_comp in zip(melt_components + plag_components, melt_sd_components + plag_sd_components):
        condition = (data[comp] == 0) & (data[sd_comp] == 0)
        data.loc[condition, comp] = 1e-6
        data.loc[condition, sd_comp] = 1e-6

    
    augmented_data = []

    for index, row in data.iterrows():
        with pm.Model() as model:
            
            melt_means = row[melt_components].values.astype(np.float64)
            melt_sds = row[melt_sd_components].values.astype(np.float64)

            plag_means = row[plag_components].values.astype(np.float64)
            plag_sds = row[plag_sd_components].values.astype(np.float64)

            
            melt_samples = []
            for i in range(len(melt_components)):
                lower_bound = max(0, melt_means[i] - 3 * melt_sds[i])  
                upper_bound = melt_means[i] + 3 * melt_sds[i] 
                initial_value = melt_means[i]  

                comp = pm.TruncatedNormal(
                    f'melt_{melt_components[i]}',
                    mu=melt_means[i],
                    sigma=melt_sds[i],
                    lower=lower_bound,
                    upper=upper_bound,
                    initval=initial_value  
                )
                melt_samples.append(comp)

            
            plag_samples = []
            for i in range(len(plag_components)):
                lower_bound = max(0, plag_means[i] - 3 * plag_sds[i])  
                upper_bound = plag_means[i] + 3 * plag_sds[i]  
                initial_value = plag_means[i]  

                comp = pm.TruncatedNormal(
                    f'plag_{plag_components[i]}',
                    mu=plag_means[i],
                    sigma=plag_sds[i],
                    lower=lower_bound,
                    upper=upper_bound,
                    initval=initial_value  
                )
                plag_samples.append(comp)

            
            sum_melt = pm.Deterministic('sum_melt', at.sum(melt_samples))
            sum_plag = pm.Deterministic('sum_plag', at.sum(plag_samples))

            
            melt_target_sum = row['SUM_melt']
            plag_target_sum = row['SUM_plag']

            
            sigma_melt = 1  
            sigma_plag = 1  

            melt_penalty = pm.Potential('melt_penalty', -((sum_melt - melt_target_sum) ** 2) / (2 * sigma_melt ** 2))
            plag_penalty = pm.Potential('plag_penalty', -((sum_plag - plag_target_sum) ** 2) / (2 * sigma_plag ** 2))

            
            try:
                trace = pm.sample(
                    1000,
                    tune=1000,
                    chains=4,
                    random_seed=42,
                    return_inferencedata=False,
                    progressbar=False
                )
            except pm.SamplingError as e:
                print(f"Sampling error for sample {index + 1}: {e}")
                continue

        
        samples_dict = {}
        for comp in melt_components:
            samples_dict[comp] = trace.get_values(f'melt_{comp}')
        for comp in plag_components:
            samples_dict[comp] = trace.get_values(f'plag_{comp}')

        samples_df = pd.DataFrame(samples_dict)

        samples_df['SAMPLE NAME'] = row['SAMPLE NAME']
        samples_df['T'] = row['T']
        samples_df['P'] = row['P']
        samples_df['H2O'] = row['H2O']
        samples_df = samples_df.sample(n=15, random_state=42).copy()  
        zero_condition = (row[melt_components + plag_components] == 1e-6)
        samples_df.loc[:, zero_condition.index[zero_condition]] = 0
        
        print(f"Sample {index + 1} generated samples:")
        print(samples_df)

        augmented_data.append(samples_df)

        print(f"Sample {index + 1}/{len(data)} has been processed.")

    augmented_df = pd.concat(augmented_data, ignore_index=True)

    output_file = 'Data/augmented_data_MCMC.xlsx'
    augmented_df.to_excel(output_file, index=False)

    print(f"Data has been saved to: {output_file}")







