# *- https://archive.ics.uci.edu/ml/datasets/Hepatitis -*

# File has no header. The fields are:
# 01. Class: DIE, LIVE
# 02. AGE: 10, 20, 30, 40, 50, 60, 70, 80
# 03. SEX: male, female
# 04. STEROID: no, yes
# 05. ANTIVIRALS: no, yes
# 06. FATIGUE: no, yes
# 07. MALAISE: no, yes
# 08. ANOREXIA: no, yes
# 09. LIVER BIG: no, yes
# 10. LIVER FIRM: no, yes
# 11. SPLEEN PALPABLE: no, yes
# 12. SPIDERS: no, yes
# 13. ASCITES: no, yes
# 14. VARICES: no, yes
# 15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
# 16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
# 17. SGOT: 13, 100, 200, 300, 400, 500,
# 18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
# 19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90
# 20. HISTOLOGY: no, yes


import pandas as pd

def read_hepatitis(path):
    categorical = [k-1 for k in [1,3,4,5,6,7,8,9,10,11,12,13,14,20]]
    integer = [k-1 for k in [2,16,17,19]]
    real = [k-1 for k in [15,18]]
    na_values = ['?']
    df=pd.read_csv(path,header=None,na_values=na_values)
    cdf = df.iloc[:,categorical]
    cdf = cdf.fillna(cdf.mode().iloc[0])
    cdf[cdf==2]='yes'
    cdf[cdf==1]='no'
    idf = df.iloc[:,integer]
    idf = idf.fillna(idf.mode().iloc[0])
    rdf = df.iloc[:,real]
    rdf = rdf.fillna(rdf.mode().iloc[0])
    return cdf, idf, rdf
