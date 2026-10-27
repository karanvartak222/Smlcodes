import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, f_oneway

# Load Dataset
df = pd.read_csv("data.csv")

print("Dataset:")
print(df)

# ---------------------------------
# 1. INDEPENDENT T-TEST
# ---------------------------------

online = df[df["Online_Class"] == "Yes"]["Marks"]
offline = df[df["Online_Class"] == "No"]["Marks"]

t_stat, t_pvalue = ttest_ind(online, offline)

print("\n--- Independent T-Test ---")
print("Online class mean:", online.mean())
print("No online class mean:", offline.mean())

print("t-statistic:", t_stat)
print("p-value:", t_pvalue)

if t_pvalue < 0.05:
    print("Reject H0: There is a significant difference in marks.")
else:
    print("Fail to reject H0: There is no significant difference in marks.")


# ---------------------------------
# 2. CHI-SQUARE TEST
# ---------------------------------

contingency_table = pd.crosstab(
    df["Gender"],
    df["Online_Class"]
)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n--- Chi-Square Test ---")
print("Contingency Table:")
print(contingency_table)

print("Chi-square statistic:", chi2)
print("p-value:", p)

if p < 0.05:
    print("Reject H0: Gender and online class participation are associated.")
else:
    print("Fail to reject H0: Gender and online class participation are independent.")


# ---------------------------------
# 3. ONE-WAY ANOVA
# ---------------------------------

self_study = df[df["Study_Method"] == "Self-study"]["Marks"]
group_study = df[df["Study_Method"] == "Group Study"]["Marks"]
online_course = df[df["Study_Method"] == "Online Course"]["Marks"]

f_stat, anova_pvalue = f_oneway(
    self_study,
    group_study,
    online_course
)

print("\n--- One-Way ANOVA ---")

print("Self-study mean:", self_study.mean())
print("Group Study mean:", group_study.mean())
print("Online Course mean:", online_course.mean())

print("F-statistic:", f_stat)
print("p-value:", anova_pvalue)

if anova_pvalue < 0.05:
    print("Reject H0: At least one study method has a significantly different mean.")
else:
    print("Fail to reject H0: No significant difference among study methods.")





Student,Gender,Study_Method,Online_Class,Marks
1,Male,Self-study,Yes,78
2,Female,Self-study,Yes,82
3,Male,Self-study,No,65
4,Female,Self-study,Yes,85
5,Male,Self-study,No,70
6,Female,Self-study,Yes,80
7,Male,Self-study,No,68
8,Female,Self-study,Yes,88
9,Male,Self-study,No,72
10,Female,Self-study,Yes,84
11,Male,Group Study,Yes,76
12,Female,Group Study,Yes,79
13,Male,Group Study,No,73
14,Female,Group Study,Yes,81
15,Male,Group Study,No,69
16,Female,Group Study,Yes,77
17,Male,Group Study,No,71
18,Female,Group Study,Yes,83
19,Male,Group Study,No,75
20,Female,Group Study,Yes,80
21,Male,Online Course,Yes,90
22,Female,Online Course,Yes,92
23,Male,Online Course,Yes,88
24,Female,Online Course,Yes,95
25,Male,Online Course,No,85
26,Female,Online Course,Yes,91
27,Male,Online Course,Yes,89
28,Female,Online Course,Yes,94
29,Male,Online Course,No,87
30,Female,Online Course,Yes,93
