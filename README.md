# Enhancing Anomaly Detection in Aircraft Data Using Privacy-Preserving Federated Learning Models

## Author
- **Chixia Ye** - *The University of Sheffield*

## Overview
Understanding the impact on anomaly detection performance when using known Federated Learning (FL) models (e.g., MOON) on aircraft flight data with a range of privacy-preserving and personalized FL principles.

## Basic Objectives

1. Define project scope.
2. Perform a literature review on the current state of the art in FL and anomaly detection both in general and in the context of aircraft flight fault data.
3. Explore an aircraft flight dataset and partition it into non-IID clients.
4. Carry out data analysis to verify & visualize the results of the changes.
5. Implement a standard personalized FL model adjustment, explore impact of degrees of personalisation on model performance.
6. Explore different choices of partitions of the flight dataset for federated learning.

## Advanced Objectives

1. Gain an understanding of the operation of an existing FL model, incorporate differential privacy & homomorphic encryption techniques.
2. Conduct ablation tests on privacy and security components to understand how including these safeguarding principles affects classification performance, computation etc. in the context of aircraft flight data.

## Gantt Chart
```mermaid
gantt
    title UGTProject Work Plan
    dateFormat  YYYY-MM-DD

    section Semester 1
    Project scope definition           :done,    des1, 2024-09-30, 3w
    Explore dataset                    :         des3, 2024-10-21, 4w
    Literature Review                  :         des4, 2024-10-21, 5w
    Data analysis & visualisations of clients :des7, 2024-11-18, 2w
    Implement an FL model & test personalisation :des8, 2024-11-25, 3w
    Explore different data partitions  :         des9, 2024-11-25, 2w

    section Semester 2
    Establish differential privacy scheme :des10, 2025-04-05, 3w
    Implement homomorphic encryption   :         des11, 2025-04-12, 3w
    Explore the impact of including the privacy-preserving techniques :des12, 2025-04-19, 3w

    section Assessment Deadlines
    Submit Aims & Objectives           :crit,    ao, 2024-10-21, 1d
    Submit Project Progress Form 1     :crit,    ppf1, 2024-11-18, 1d
    Submit Interim Report              :crit,    ir, 2024-12-09, 1d
    Interview with Second Reader       :crit,    isr, 2025-01-13, 1d
    Submit Project Progress Form 2     :crit,    ppf2, 2025-05-12, 1d
    Submit Dissertation                :crit,    dis, 2025-06-16, 1d
    Oral Presentation                  :crit,    op, 2025-06-18, 1d

