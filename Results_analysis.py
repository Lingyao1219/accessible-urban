import pandas as pd
from functools import reduce


def format_table2(filename=' Restaurant _Gam_Pop_Pct_scale.csv', Yname='Case-fatality ratio'):
    All_corr1 = pd.read_csv(r'D:\\Google_Review\\Accessible\\' + filename)
    All_corr1['conf.low'] = All_corr1['Estimate'] - 1.96 * All_corr1['Std..Error']
    All_corr1['conf.high'] = All_corr1['Estimate'] + 1.96 * All_corr1['Std..Error']
    # All_corr1 = All_corr1.replace({"names": di_replace_nn})
    All_corr1['Symbol_'] = ' '
    All_corr1.loc[All_corr1['Pr...t..'] <= 0.001, 'Symbol_'] = '***'
    All_corr1.loc[(All_corr1['Pr...t..'] <= 0.01) & (All_corr1['Pr...t..'] > 0.001), 'Symbol_'] = '**'
    All_corr1.loc[(All_corr1['Pr...t..'] <= 0.05) & (All_corr1['Pr...t..'] > 0.01), 'Symbol_'] = '*'
    # All_corr1.loc[(All_corr1['Pr...t..'] <= 0.1) & (All_corr1['Pr...t..'] > 0.05), 'Symbol_'] = '.'
    for jj in list(All_corr1.columns):
        try:
            All_corr1[jj] = All_corr1[jj].round(3).map('{:.3f}'.format).astype(str)  # .apply('="{}"'.format)
        except:
            All_corr1[jj] = All_corr1[jj]
    n_cc = ['Adj_R2', 'ti(Lat,Lng)', 's(CBSA)', 'dev.expl', 'n']
    All_corr1.loc[~All_corr1['names'].isin(n_cc), 'Estimate'] = \
        (All_corr1.loc[~All_corr1['names'].isin(n_cc), 'Estimate'] + All_corr1.loc[
            ~All_corr1['names'].isin(n_cc), 'Symbol_']
         # + '\n(' + All_corr1.loc[
         #     ~All_corr1['names'].isin(n_cc), 'conf.low'] + ', ' + \
         # All_corr1.loc[~All_corr1['names'].isin(n_cc), 'conf.high'] + ')'
         )

    # Return
    All_corr_final = All_corr1[['names', 'Estimate']]
    # All_corr_final.to_excel(r'D:\\Vaccination\\Results\\' + outname)
    All_corr_final.columns = ['Variables', Yname]
    All_corr_final = All_corr_final[All_corr_final['Variables'] != 'Blank']
    return All_corr_final


di_replace_nn \
    = {"Asian_R": "Asian", 'Black_Non_Hispanic_R': "African American", 'Democrat_R': 'Democrat',
       'HISPANIC_LATINO_R': 'Hispanic', 'Indian_R': 'Others', 'Employment_Density': 'Employment Density',
       "Household_Below_Poverty_R": 'Poverty', "Male_R": "Male", 'Bt_45_64_R': 'Age 45-64',
       "Rural_Population_R": "Rural Population", 'Transit_Freq': 'Transit Frequency', 'Zero_car_R': 'Zero Car',
       'Two_plus_car_R': '>=2 Cars', 'One_car_R': 'One Car', 'Parking_poi_density': 'Parking POI Density',
       "Median_income": "Median Income", 'avg_poi_score': 'Avg. POI Score', 'Road_Density': 'Road Density',
       'Urbanized_Areas_Population_R': 'Urban Population', "Over_65_R": "Age over 65", "Under_18_R": "Age under 18",
       'Education_Degree_R': 'Highly-Educated', 'White_Non_Hispanic_R': 'White',
       "Population_Density": "Population Density", "Transit_Freq_Area": "Transit Frequency",
       "CategoriesApartment": "Apartment", "CategoriesHotel": "Hotel", "CategoriesPersonal Service": "Personal Service",
       "CategoriesRecreation": "Recreation", "CategoriesRestaurant": "Restaurant",
       "CategoriesRetail Trade": "Retail Trade", "Adj_R2": 'Adj. R2'}

gam_all = format_table2(filename=' access_ct _Gam_scale.csv', Yname='All')
cross_coeff_all = gam_all.replace({"Variables": di_replace_nn})
cross_coeff_all.to_excel(r'D:\\Google_Review\\Accessible\\Gam_all_cross_coeff_ct.xlsx')
gam_all = format_table2(filename=' access_bg _Gam_scale.csv', Yname='All')
cross_coeff_all = gam_all.replace({"Variables": di_replace_nn})
cross_coeff_all.to_excel(r'D:\\Google_Review\\Accessible\\Gam_all_cross_coeff_bg.xlsx')
