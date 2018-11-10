# tb-exosomal-mirna
data and code for article (Integrating exosomal microRNA and electronic health data improves tuberculosis diagnosis)

Data Description: 

The data fold includes four subfolders: ‘tbm-vs-health’, ‘tbm-vs-dc’, ‘ptb-vs-health’ and ‘ptb-vs-dc’; and these subfolders are used for the modelling for TBM patients vs HS controls, TBM patients vs TBM-DC patients, PTB patients vs HS controls, and PTB patients vs PTB-DC patients, respectively. In each subfolder, the file train_df.csv and test_df.csv are the normalized data for modeling in the selection cohort and that in the testing cohort, respectively. 

Abbreviation in CSV file: Hct, hematocrit; PLT, platelets; WBC, white blood cell counts;  N, neutrophils; L, lymphocytes; M, monocytes; Alb, albumin. RBC, red blood cell counts; Hb, hemoglobin.

Example: 

python main.py --data-folder data/ptb-vs-dc/ --positive-label PTB
