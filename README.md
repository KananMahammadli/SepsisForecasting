# Early Sepsis Detection Through Time-Aware Embedding and Temporal Short-term Memory-based LSTM
Early sepsis detection is crucial as it has a 6 million per year mortality rate, and it is one of the costliest medical conditions. Yet, it is still challenging to forecast sepsis onset due to the lack of a general methodology for identifying dynamic patterns of the syndrome. Deep learning approaches, especially recurrent neural networks-based architectures, have been explored for automatic sepsis detection, considering fixed 4, 6, and 8 hours before sepsis onset. As the last 6 hours before sepsis onset can have life-threatening ramifications, it is more important to forecast sepsis till the 6th hour in a multistep-ahead manner. This study aims to test the hypothesis that sepsis onset can be determined before crucial hours by focusing on data enrichment and custom modeling methodology for sepsis-specific data patterns. This paper suggests: 
- **Data cleaning pipeline and feature engineering** through masking, lagging, rolling window, and delta features for data enrichment
- **Time-aware embeddings and temporal feature-based short-term memory** to capture irregularities in data
- **Customized loss function** to handle imbalanced sequential learning

The system achieves an F1-score of 0.097 on validation data and 0.116 on test data, close to current short-term sepsis detection scores in the literature, around 0.1.

## Model Training Modes
The LSTM model has been trained in four different modes, with the latest mode representing the main architecture used in this study, while the first three modes provided valuable intermediate results through research:

1. **Mode 1**: Trained using only vital signs and demographic data, without any feature engineering, embeddings, or convolution.
2. **Mode 2**: Adds engineered features (e.g., lagging and delta features) to the previous structure.
3. **Mode 3**: Builds upon Mode 2 by incorporating embeddings for LSTM inputs.
4. **Mode 4**: Enhances Mode 3 with convolution-based short-term memory extraction, replacing the initial hidden state of LSTM with newly created memory, and integrating all previous improvements.

## Model Architecture:
![image](https://github.com/user-attachments/assets/6a7a0bb1-f4dd-4066-b4a0-9ed19d322d02)

## Custom Weighted Loss:
![image](https://github.com/user-attachments/assets/5b7365be-7341-4095-bae0-885ce88a5c7a)

## Results:
![image](https://github.com/user-attachments/assets/4cb590c3-da81-4866-9220-c14ea8b5cace)
