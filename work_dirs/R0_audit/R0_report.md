# R0 Data Preparation Report

## 1. Dataset Overview

Dataset:
OpenEarthMap

Task:
Remote sensing semantic segmentation

R0 stage:
Dataset audit and region split preparation.


## 2. Data Audit

Completed scripts:

- R0_data_audit.py
- R0_metadata.py
- R0_label_audit.py


Audit items:

- Image-label pair consistency
- Dataset metadata statistics
- Label class distribution


Result:

All checked regions contain valid image-label pairs.
No missing image files were detected.


## 3. Sample Visualization

Script:

R0_sample_visualization.py


Purpose:

Visual inspection of remote sensing images and corresponding labels.


Result:

Sample visualization was successfully generated.


Output:

sample_visualization.png


## 4. Region Split Preparation

Script:

R0_prepare_region_split.py


Purpose:

Prepare candidate regions for spatial split.


Dataset statistics:

- Total regions: 97
- Valid regions with image-label pairs: 75


OOD candidate selection:

Approximately 15%-20% regions were considered as OOD test candidates.


## 5. Region Coverage Analysis

Script:

R0_analyze_region_coverage.py


Analysis:

- Number of semantic classes in each region
- Class coverage completeness


Result:

The dataset contains:

- Class IDs: 0-8
- Total classes: 9


Most candidate regions contain complete semantic categories.


## 6. Region Split Review

Script:

R0_review_region_split.py


Review criteria:

- Number of valid samples
- Number of classes
- Class completeness


Summary:

OOD candidate regions were reviewed.

Candidate regions:

19


Regions containing all classes:

16


Regions with incomplete classes:

3


## 7. Conclusion

The R0 data preparation stage has been completed.

The dataset has been audited and analyzed.

Prepared outputs can be used for:

- Train/Validation/Test split
- SegFormer training configuration
- Further semantic segmentation experiments