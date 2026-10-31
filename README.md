# pulmoTrack-farahsayari contribution

Pulmonary Nodule Segmentation – LUNA16 Dataset

## Project Overview
This project focuses on improving segmentation of pulmonary nodules—especially **tiny nodules**—using deep learning methods on the **LUNA16 dataset**. The goal is to push the boundaries of current models by enhancing performance where it is typically weakest: detecting small, hard-to-segment nodules in 3D CT scans.

##  Key Contributions

-  Implemented **3D Attention U-Net** and **Residual Attention U-Net** architectures.
-  Applied **patch-based training** to manage large 3D CT volumes efficiently.
-  Performed extensive **data augmentation** (flip, rotate, zoom, intensity shift, noise).
-  Introduced **oversampling strategies** to emphasize tiny nodules in the training process.
-  Evaluated model performance with focus on **tiny vs large nodules**.

>  While large nodules were segmented with high accuracy, **tiny nodules remain a significant challenge**, confirming limitations of current datasets and model capacities.

##  Comparative Performance

According to the systematic review by **Gao et al. (2025)**, various state-of-the-art studies reported Dice scores on LUNA16 ranging from **0.77 to 0.93**. Our results show that:
- The model performs **competitively** for medium and large nodules.
- With a **validation DSC of 0.95**, our architecture aligns with or exceeds most reported methods—though tiny nodules still underperform, highlighting an area for future work.

---

*Maintained by Farah Sayari*

*Last Updated: May 2025*
