# Disentangling Interest and Conformity for an Adversarial Learning-Based Personalized Recommendation System

Welcome to the code repository for the DICA project.

🚀 About the Project

The [DICA](https://github.com/tsinghua-fib-lab/DICE) model addresses the challenges of popularity bias in recommendation systems, which often overemphasize popular items

![이미지 2025  1  13  오후 9 36](https://github.com/user-attachments/assets/99334f9a-7039-4cf5-b2aa-15537ec78b95)

This project aims to:

+ Rank learning with DICE

    We utilize the DICE mechanism to learn rankings, which employs disentangled embeddings of conformity and interest.
---
+ Adversarial learning

    Additionally, adversarial learning is employed to promote fair recommendations between popular and unpopular items.
---

Data

We used the Movielens-10M and Netflix datasets. To convert explicit feedback data into implicit feedback data, we used only the data with a rating of 5 as an indicator of positive feedback.

<div align="center">

| **Dataset** | **User** | **Item** | **Interaction** |
|-------------|----------|----------|-----------------|
| Movielens-10M | 37,962   | 4,819    | 1,371,473       |
| Netflix       | 32,450   | 8,432    | 2,212,690       |

---
</div>



Experiment result
<div align="center">
<img src="https://github.com/user-attachments/assets/b57ed16c-2384-4030-b71f-35f829d2ec35" alt="Image" width="600" />
</div>
We experienced a slight decrease in performance in terms of accuracy metrics(Recall, HR, NDCG), but achieved significant improvements in fairness metrics(RSP, REO). Notably, we achieved an over 10% improvement in the RSP metric.






<div align="center">
<img src="https://github.com/user-attachments/assets/4b668f4d-2c41-498d-a32d-17ca38ae525a" alt="Image" width="600" />
<img src="https://github.com/user-attachments/assets/c8f04fd1-9ace-4200-bdfc-57f2ef4d953c" alt="Image" width="600" />
</div>

This graph shows the interaction score distribution for combinations of user and item groups, categorized by popularity. The overlap of all four graph types indicates that the proposed model fairly recommends items from both popular and less popular groups to both user types.
