import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import scipy
from scipy.stats import t
from scipy import stats

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': False, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})
pd.options.mode.chained_assignment = None


def welch_ttest(x1, x2):
    n1 = x1.size
    n2 = x2.size
    v1 = np.var(x1, ddof=1)
    v2 = np.var(x2, ddof=1)
    pooled_se = np.sqrt(v1 / n1 + v2 / n2)
    delta = stats.trim_mean(x1, 0.01) - stats.trim_mean(x2, 0.01)
    tstat = delta / pooled_se
    df = (v1 / n1 + v2 / n2) ** 2 / (v1 ** 2 / (n1 ** 2 * (n1 - 1)) + v2 ** 2 / (n2 ** 2 * (n2 - 1)))
    # two side t-test
    p = 2 * t.cdf(-abs(tstat), df)
    # upper and lower bounds
    lb = delta - t.ppf(0.975, df) * pooled_se
    ub = delta + t.ppf(0.975, df) * pooled_se
    return pd.DataFrame(np.array([tstat, df, p, delta, lb, ub]).reshape(1, -1),
                        columns=['T statistic', 'df', 'pvalue 2 sided', 'Difference in mean', 'lb', 'ub'])


di_replace_nn \
    = {"Asian_R": "Asian", 'Black_Non_Hispanic_R': "African American", 'Democrat_R': 'Democrat',
       'HISPANIC_LATINO_R': 'Hispanic', 'Indian_R': 'Others', 'Employment_Density': 'Employment Density',
       "Household_Below_Poverty_R": 'Below Poverty', "Male_R": "Male", 'Bt_45_64_R': 'Age 45-64',
       'Bt_18_44_R': 'Age 18-44', "Rural_Population_R": "Rural Population", 'Transit_Freq': 'Transit Frequency',
       'Zero_car_R': 'Zero Car', 'Two_plus_car_R': '>=2 Cars', 'One_car_R': 'One Car',
       'Parking_poi_density': 'Parking POI Density', "Median_income": "Median Income",
       'avg_poi_score': 'Avg. POI Score', 'Road_Density': 'Road Density',
       'Urbanized_Areas_Population_R': 'Urban Population', "Over_65_R": "Age over 65", "Under_18_R": "Age under 18",
       'Education_Degree_R': 'Highly-Educated', 'White_Non_Hispanic_R': 'White',
       "Population_Density": "Population Density", "Total_poi_density": 'POI Density'}

# Read MSA Geo data
MSA_geo = gpd.GeoDataFrame.from_file(r'D:\Google_Review\Parking\tl_2019_us_cbsa\tl_2019_us_cbsa.shp')
MSA_geo = MSA_geo.to_crs(epsg=5070)
MSA_geo.rename({'CBSAFP': 'CBSA'}, axis=1, inplace=True)
# print(len(set(MSA_geo['CSAFP'])))  # CSAFP means CSA

# Read CBG Geo data
poly = pd.read_pickle(r'D:\Google_Review\Parking\temp\poly_5070.pkl')

# Read other features
CT_Features = pd.read_csv(r'F:\Research_Old\COVID19-Socio\Data\CBG_COVID_19.csv', index_col=0)
CT_Features['BGFIPS'] = CT_Features['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
smart_loc = pd.read_pickle(r'F:\Research_Old\Incentrip_research\data\SmartLocationDatabaseV3\SmartLocationDatabase.pkl')
msa_population = smart_loc[['CBSA', 'CBSA_POP', 'CBSA_Name']].drop_duplicates(subset=['CBSA']).reset_index(drop=True)
poi_ty = pd.read_excel(r'D:\Google_Review\Parking\temp\places_summary_hsh.xlsx')
poi_ty = poi_ty[['naics_code', 'Categories']]

'''
# Read sentiment data
all_files = glob.glob(r'D:\Google_Review\Accessible\accessible-poi-metrics\*')
g_pois = pd.DataFrame()
for kk in all_files:
    e_json = pd.read_json(kk, lines=True)
    g_pois = pd.concat([g_pois, e_json], ignore_index=True)
# g_pois = pd.read_pickle(r'D:\Google_Review\Parking\temp\park_all_new.pkl')
g_pois = gpd.GeoDataFrame(g_pois, geometry=gpd.points_from_xy(g_pois['longitude'], g_pois['latitude']))
g_pois = g_pois.set_crs('EPSG:4326')
g_pois = g_pois.to_crs('EPSG:5070')
SInUS = gpd.sjoin(g_pois, poly, how='inner', predicate='within').reset_index(drop=True)
SInUS = SInUS[['gmap_id', 'BGFIPS']]
SInUS = SInUS.drop_duplicates().reset_index(drop=True)
g_poisf = g_pois.merge(SInUS, on='gmap_id')
g_poisf.to_pickle(r'D:\Google_Review\Accessible\g_poisf.pkl')
'''

g_pois = pd.read_pickle(r'D:\Google_Review\Accessible\g_poisf.pkl')
g_pois['BGFIPS'] = g_pois['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
g_pois = g_pois.merge(poi_ty, on='naics_code')
g_pois['Categories'] = g_pois['Categories'].fillna('Others')
g_pois['Categories'].value_counts()
g_pois['total_accessible_reviews'] = g_pois['total_accessible_reviews'].astype(int)
g_pois['avg_accessible_sentiment'] = g_pois['avg_accessible_sentiment'].astype(float)
g_pois['sum_comment'] = g_pois.groupby(['BGFIPS'])['total_accessible_reviews'].transform("sum")
g_pois['weight_st'] = (g_pois['total_accessible_reviews'] / g_pois['sum_comment']) * g_pois['avg_accessible_sentiment']
sns.displot(g_pois['weight_st'])

# Average sentiment to CBG
bg_pois = g_pois.groupby(['BGFIPS'])['weight_st'].sum().reset_index()
bg_count = g_pois.groupby(['BGFIPS'])['total_accessible_reviews'].sum().reset_index()
bg_pois = bg_pois.merge(bg_count, on=['BGFIPS'])
bg_pois = bg_pois[bg_pois['BGFIPS'].isin(poly['BGFIPS'])].reset_index(drop=True)

CT_Features = CT_Features.merge(
    smart_loc[['BGFIPS', 'D3B', 'Pct_AO0', 'Pct_AO1', 'Pct_AO2p', 'NatWalkInd', 'D3A', 'D1C', 'D4C', 'CBSA', 'CBSA_POP',
               'Ac_Total', 'CSA', 'D4A', 'D4D', 'D4E']], on='BGFIPS')
# Rename
CT_Features = CT_Features.rename(
    columns={'D3B': 'Intersection_Density', 'Pct_AO0': 'Zero_car_R', 'Pct_AO1': 'One_car_R', 'D4E': 'Transit_Freq_Pop',
             'Pct_AO2p': 'Two_plus_car_R', 'NatWalkInd': 'Walkability', 'D3A': 'Road_Density',
             'D1C': 'Employment_Density', 'D4C': 'Transit_Freq', 'D4A': 'Distance_Transit', 'D4D': 'Transit_Freq_Area'})
CT_Features.loc[CT_Features['Transit_Freq'] < 0, 'Transit_Freq'] = 0
CT_Features.loc[CT_Features['Transit_Freq_Area'] < 0, 'Transit_Freq_Area'] = 0
CT_Features.loc[CT_Features['Transit_Freq_Pop'] < 0, 'Transit_Freq_Pop'] = 0
CT_Features.loc[CT_Features['Distance_Transit'] < 0, 'Distance_Transit'] = 1207.008

# read disability
disability = pd.read_csv(r'D:\Google_Review\Accessible\nhgis0017_csv\nhgis0017_ds262_20225_blck_grp.csv',
                         encoding='latin-1')
disability = disability[
    ['GISJOIN', 'AQR0E001', 'AQR0E005', 'AQR0E008', 'AQR0E012', 'AQR0E015', 'AQR0E020', 'AQR0E023', 'AQR0E027',
     'AQR0E030']]
disability['BGFIPS'] = disability['GISJOIN'].str[1:3] + disability['GISJOIN'].str[4:7] + \
                       disability['GISJOIN'].str[8:14] + disability['GISJOIN'].str[14:15]
disability['Disability_below_poverty_R'] = 100 * (
        disability['AQR0E005'] + disability['AQR0E012'] + disability['AQR0E020'] +
        disability['AQR0E027']) / disability['AQR0E001']
disability['Disability_above_poverty_R'] = 100 * (
        disability['AQR0E008'] + disability['AQR0E015'] + disability['AQR0E023'] +
        disability['AQR0E030']) / disability['AQR0E001']
disability = disability[['Disability_below_poverty_R', 'Disability_above_poverty_R', 'BGFIPS']]
disability['Disability'] = disability['Disability_below_poverty_R'] + disability['Disability_above_poverty_R']

CT_Features = CT_Features.merge(disability, on='BGFIPS', how='left')
CT_Features = CT_Features.fillna(0)

# Drop outliers
outliers_ns = []
for kk in ['Black_Non_Hispanic_R', 'HISPANIC_LATINO_R', 'Population_Density', 'Employment_Density',
           'Household_Below_Poverty_R', 'Over_65_R', 'Education_Degree_R', 'Transit_Freq_Area', 'Zero_car_R', 'Male_R',
           'One_car_R', 'Bt_45_64_R']:
    outliers_n = CT_Features.loc[(CT_Features[kk] > np.percentile(CT_Features[kk], 99.9)) |
                                 (CT_Features[kk] < np.percentile(CT_Features[kk], 0.1)), 'BGFIPS'].to_list()
    print(kk + ': %s' % len(outliers_n))
    outliers_ns += outliers_n
len(set(outliers_ns))

# Merge with POI info
g_fe = pd.read_pickle(r'D:\Google_Review\Parking\temp\g_fe.pkl')
bg_pois = bg_pois.merge(g_fe[['BGFIPS', 'num_of_reviews', 'total_poi_count', 'avg_poi_score']],
                        on='BGFIPS').reset_index(drop=True)
bg_pois_raw = bg_pois.copy()

# Merge with CBG features
bg_pois = bg_pois.merge(CT_Features, on='BGFIPS').reset_index(drop=True)
# Merge with MSA features
bg_pois = bg_pois.merge(MSA_geo[['CBSA', 'NAME', 'NAMELSAD', 'LSAD']], on='CBSA')
# Polish data
bg_pois['Total_review_density'] = bg_pois['num_of_reviews'] / bg_pois['ALAND']
bg_pois['Total_poi_density'] = bg_pois['total_poi_count'] / bg_pois['ALAND']
bg_pois['Accessible_review_density'] = bg_pois['total_accessible_reviews'] / bg_pois['ALAND']
bg_pois['Median_income'] = bg_pois['Median_income'] / 1e4
bg_pois['Population_Density'] = bg_pois['Population_Density'] / 640  # to acre
# Only greater than 10
bg_pois = bg_pois[(bg_pois['total_accessible_reviews'] > 10)].reset_index(drop=True)
bg_pois = bg_pois[~bg_pois['BGFIPS'].isin(outliers_ns)].reset_index(drop=True)
bg_pois = bg_pois.fillna(0)
# sns.displot(bg_pois['weight_st'])  # .loc[bg_pois['Categories'] == 'Hotel', 'weight_st']


# All CBG: Merge with CBG features
bg_pois_raw = bg_pois_raw.merge(CT_Features, on='BGFIPS', how='right').reset_index(drop=True)
# Polish data
bg_pois_raw['Total_review_density'] = bg_pois_raw['num_of_reviews'] / bg_pois_raw['ALAND']
bg_pois_raw['Total_poi_density'] = bg_pois_raw['total_poi_count'] / bg_pois_raw['ALAND']
bg_pois_raw['Accessible_review_density'] = bg_pois_raw['total_accessible_reviews'] / bg_pois_raw['ALAND']
bg_pois_raw['Median_income'] = bg_pois_raw['Median_income'] / 1e4
bg_pois_raw['Population_Density'] = bg_pois_raw['Population_Density'] / 640  # to acre
# Only greater than 10
# bg_pois = bg_pois[(bg_pois['total_accessible_reviews'] > 10)].reset_index(drop=True)
bg_pois_raw = bg_pois_raw[~bg_pois_raw['BGFIPS'].isin(outliers_ns)].reset_index(drop=True)

# Output data: To R for modelling
# bg_pois = bg_pois.dropna(subset=['LSAD']).reset_index(drop=True)
need_scio = ['Population_Density', 'Bt_18_44_R', 'Asian_R', 'Over_65_R', 'Republican_R', 'Intersection_Density',
             'Bt_45_64_R', 'Urbanized_Areas_Population_R', 'Indian_R', 'Rural_Population_R',
             'Household_Below_Poverty_R', 'Employment_Density', 'Two_plus_car_R', 'HISPANIC_LATINO_R', 'Zero_car_R',
             'Black_Non_Hispanic_R', 'CBSA_POP', 'Total_Population', 'White_Non_Hispanic_R', 'Male_R',
             'Urban_Clusters_Population_R', 'Road_Density', 'Walkability', 'One_car_R', 'Transit_Freq', 'Disability',
             'Education_Degree_R', 'Democrat_R', 'Median_income', 'Transit_Freq_Area', 'Accessible_review_density',
             'Transit_Freq_Pop', 'num_of_reviews', 'total_poi_count', 'avg_poi_score', 'total_accessible_reviews',
             "Total_review_density", "Total_poi_density", 'Disability_below_poverty_R', 'Disability_above_poverty_R']
fbg_pois = bg_pois[['BGFIPS', 'weight_st', 'ALAND', 'Lng', 'Lat', 'CBSA', 'LSAD'] + need_scio]
fbg_pois.to_csv(r'D:\Google_Review\Accessible\bg_senti_access.csv')
fbg_pois.isnull().sum()
fbg_pois.corr(numeric_only=True).to_csv(r'D:\Google_Review\Accessible\bg_access_corr.csv')
fbg_pois.describe().T.to_csv(r'D:\Google_Review\Accessible\bg_access_des.csv')


# Plot scatter
def plot_scatter(poi_msa, x_t, x_name, cck):
    fig, ax = plt.subplots(figsize=(5, 5))
    # poi_msa = bg_pois[(bg_pois['Categories'] == poi_t)].reset_index(drop=True)
    poi_msa['size'] = (poi_msa['Total_Population'] / max(poi_msa['Total_Population']) * 800)
    poi_msa.loc[poi_msa['size'] < 5, 'size'] = 5
    # poi_msa = poi_msa[(poi_msa[x_t] < np.percentile(poi_msa[x_t], 99)) &
    #                   (poi_msa[x_t] > np.percentile(poi_msa[x_t], 1))].reset_index(drop=True)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x=poi_msa[x_t], y=poi_msa['weight_st'])
    sns.regplot(x=poi_msa[x_t], y=(poi_msa['weight_st']), color=cck,
                scatter_kws={'s': (poi_msa['size']), 'alpha': 0.6}, ax=ax)
    ax.set_ylabel('Sentiment')
    ax.set_xlabel(x_name)
    # plt.title(poi_t)
    plt.text(0.65, 0.85, 'Y = %sX + %s\n(Pearson = %s)' % (round(slope, 3), round(intercept, 3), round(r_value, 3)),
             horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(r'D:\Google_Review\Accessible\scatter_plot_%s_bg.pdf' % (x_t))
    plt.close()


plot_scatter(bg_pois, 'Disability', "Disability (%)", 'royalblue')
plot_scatter(bg_pois, 'Disability_below_poverty_R', "Disability below poverty (%)", 'royalblue')
plot_scatter(bg_pois, 'Disability_above_poverty_R', "Disability above poverty (%)", 'royalblue')

# Plot spatial map: by POI type, total us
bg_pois_geo = poly.merge(bg_pois.loc[:, ['BGFIPS', 'weight_st', 'CBSA']], on='BGFIPS')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
bg_pois_geo.plot(column='weight_st', ax=ax, legend=True, scheme='user_defined',
                 classification_kwds={'bins': [-0.5, 0, 0.5]}, cmap='bwr', k=6,
                 legend_kwds=dict(frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0)), linewidth=0,
                 edgecolor='white', alpha=0.8)  # natural_breaks
(MSA_geo[(MSA_geo['LSAD'] == 'M1') & (MSA_geo['CBSA'].isin(bg_pois_geo['CBSA']))]
 .geometry.boundary.plot(color=None, edgecolor='k', linewidth=0.5, alpha=0.8, ax=ax))
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
ax.axis('off')
plt.tight_layout()
plt.savefig(r'D:\Google_Review\Accessible\spatial_sentiment_cbg.pdf')
plt.close()

# fbg_pois.loc[fbg_pois['total_accessible_reviews']>10,'Disability_below_poverty_R'].mean()
# fbg_pois.loc[fbg_pois['total_accessible_reviews']<=10,'Disability_below_poverty_R'].mean()

# Plot corr
all_poi_corr = fbg_pois.corr(numeric_only=True)
all_poi_corr = all_poi_corr['weight_st']
# all_poi_corr = all_poi_corr[all_poi_corr['variable'].isin(bg_pois['Categories'].value_counts().head(6).index)]
all_poi_corr = all_poi_corr.sort_values(ascending=False, key=abs)
all_poi_corr = all_poi_corr[all_poi_corr < 0.999]
fig, ax = plt.subplots(figsize=(6, 7))
sns.barplot(data=all_poi_corr.head(15), ax=ax, orient='h')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
plt.legend(loc=2)
plt.xlabel('Correlation')
plt.tight_layout()
plt.savefig(r'D:\Google_Review\Accessible\corr_ct_access.pdf')

# Plot comparison: acc vs no acc
need_scio = ['Population_Density', 'Bt_18_44_R', 'Asian_R', 'Over_65_R',
             'Bt_45_64_R', 'Urbanized_Areas_Population_R', 'Indian_R', 'Rural_Population_R',
             'Household_Below_Poverty_R', 'Employment_Density', 'HISPANIC_LATINO_R',
             'Black_Non_Hispanic_R', 'Total_Population', 'White_Non_Hispanic_R', 'Male_R',
             'Road_Density', 'Disability', 'Education_Degree_R', 'Median_income']

bg_pois_raw['is_access'] = True
bg_pois_raw.loc[bg_pois_raw['total_accessible_reviews'] < 1, 'is_access'] = False
All_Ttest = pd.DataFrame()
for yvar in need_scio:
    atest = bg_pois_raw.loc[bg_pois_raw['is_access'], yvar]
    btest = bg_pois_raw.loc[~bg_pois_raw['is_access'], yvar]
    ttest_resu = welch_ttest(btest.dropna(), atest.dropna())
    ttest_resu['Yvar'] = yvar
    ttest_resu['Mean'] = bg_pois_raw[yvar].mean()
    ttest_resu['SD'] = bg_pois_raw[yvar].std()
    All_Ttest = pd.concat([All_Ttest, ttest_resu])
All_Ttest['pct'] = -(All_Ttest['Difference in mean'] / All_Ttest['Mean']) * 100
All_Ttest = All_Ttest.replace({"Yvar": di_replace_nn})
All_Ttest = All_Ttest.sort_values(by='pct', ascending=False)

fig, ax = plt.subplots(figsize=(6, 5))
sns.barplot(data=All_Ttest, x='pct', y='Yvar', palette='coolwarm', ax=ax)
ax.xaxis.grid(True)
# plt.title('All POIs')
plt.xlabel('Difference (%): With vs. W/O Reviews')
plt.ylabel('')
for p in ax.patches:
    ax.annotate("%.1f" % p.get_width() + '%', xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                xytext=(5, 0), textcoords='offset points', ha="left", va="center", fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig(r'D:\Google_Review\Accessible\diff_cbg.pdf')
