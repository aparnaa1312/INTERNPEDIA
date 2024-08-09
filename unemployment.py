import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file1_df=pd.read_csv('C://Unemployment in India.csv')
file2_df=pd.read_csv('C:\\Unemployment_Rate_upto_11_2020.csv')
file2_df.rename(columns={'Region.1':'region_group'},inplace=True)
file1_df.columns=file1_df.columns.str.strip()
file2_df.columns=file2_df.columns.str.strip()
file1_df['Date']=file1_df['Date'].str.strip()
file2_df['Date']=file2_df['Date'].str.strip()
file1_df['Date'] = pd.to_datetime(file1_df['Date'], format='%d-%m-%Y',dayfirst=True)
file2_df['Date'] = pd.to_datetime(file2_df['Date'], format='%d-%m-%Y',dayfirst=True)
cc=['Region', 'Date', 'Frequency', 'Estimated Unemployment Rate (%)','Estimated Employed', 'Estimated Labour Participation Rate (%)']
combined=pd.merge(file1_df,file2_df,how='outer',on=cc)
numeric_columns=combined.select_dtypes(include='number').columns
combined[numeric_columns]=combined[numeric_columns].fillna(combined[numeric_columns].mean())
object_columns=combined.select_dtypes(exclude='number').columns
for i in object_columns:
    if i!='Date':
        combined[i]=combined[i].fillna('unknown')
combined=combined.dropna(subset=['Date'])
combined=combined.sort_values(by='Date',ascending=True)
print("combined dataframe info",combined.info())
print("first few rows",combined.head())
print("descriptive stat",combined.describe())
#average_umeployement_rate_over_time
unemployment_rate_over_time = combined.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
plt.figure(figsize=(12, 6))
plt.plot(unemployment_rate_over_time, marker='o', linestyle='-', color='b')
plt.title('Average Estimated Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.grid(True)
plt.show()
plt.figure(figsize=(12,6))
sns.boxplot(x='Region',y='Estimated Unemployment Rate (%)',data=combined)
plt.title('unemployement rate by region')
plt.xlabel('region')
plt.ylabel('unemployement rate')
plt.xticks(rotation=90)
plt.show()











