"""Model metrics data extracted from original templates."""

MODEL_METRICS = {
    # Mammography - ResNet
    ('mammography', 'resnet', 'initial'): {
        'metrics': """Accuracy   : 0.9782
Precision  : 0.9227
Recall     : 0.9078
F1-score   : 0.9152
ROC-AUC    : 0.9949""",
        'report': """               precision    recall  f1-score   support

           0     0.9863    0.9887    0.9875      2920
           1     0.9227    0.9078    0.9152       434

    accuracy                         0.9782      3354
   macro avg     0.9545    0.9483    0.9514      3354
weighted avg     0.9781    0.9782    0.9782      3354""",
        'about': "This model is based on the ResNet50 architecture, specifically designed for mammography image classification. It has been trained on a dataset of mammogram images to distinguish between normal and abnormal cases. The model achieves high accuracy and robust performance metrics, making it suitable for clinical applications in breast cancer detection."
    },
    ('mammography', 'resnet', 'finetuned'): {
        'metrics': """Accuracy   : 0.9640
Precision  : 0.8812
Recall     : 0.8479
F1-score   : 0.8643
ROC-AUC    : 0.9872""",
        'report': """               precision    recall  f1-score   support

           0     0.9789    0.9856    0.9822      2920
           1     0.8812    0.8479    0.8643       434

    accuracy                         0.9640      3354
   macro avg     0.9301    0.9168    0.9232      3354
weighted avg     0.9636    0.9640    0.9637      3354""",
        'about': "This model is based on the ResNet50 architecture with fine-tuning, specifically designed for mammography image classification."
    },
    # Mammography - VGG
    ('mammography', 'vgg', 'initial'): {
        'metrics': """Accuracy   : 0.9672
Precision  : 0.8750
Recall     : 0.8710
F1-score   : 0.8730
ROC-AUC    : 0.9889""",
        'report': """               precision    recall  f1-score   support

           0     0.9808    0.9815    0.9812      2920
           1     0.8750    0.8710    0.8730       434

    accuracy                         0.9672      3354
   macro avg     0.9279    0.9262    0.9271      3354
weighted avg     0.9671    0.9672    0.9672      3354""",
        'about': "This model is based on the VGG16 architecture, specifically designed for mammography image classification."
    },
    ('mammography', 'vgg', 'finetuned'): {
        'metrics': """Accuracy   : 0.9559
Precision  : 0.8664
Recall     : 0.7785
F1-score   : 0.8201
ROC-AUC    : 0.9806""",
        'report': """               precision    recall  f1-score   support

           0     0.9676    0.9822    0.9748      4380
           1     0.8664    0.7785    0.8201       650

    accuracy                         0.9559      5030
   macro avg     0.9170    0.8803    0.8975      5030
weighted avg     0.9545    0.9559    0.9548      5030""",
        'about': "This model is based on the VGG16 architecture with fine-tuning, specifically designed for mammography image classification."
    },
    # Mammography - DenseNet
    ('mammography', 'densenet', 'initial'): {
        'metrics': """Accuracy   : 0.9714
Precision  : 0.8789
Recall     : 0.9032
F1-score   : 0.8909
ROC-AUC    : 0.9931""",
        'report': """               precision    recall  f1-score   support

           0     0.9856    0.9815    0.9835      2920
           1     0.8789    0.9032    0.8909       434

    accuracy                         0.9714      3354
   macro avg     0.9322    0.9424    0.9372      3354
weighted avg     0.9718    0.9714    0.9715      3354""",
        'about': "This model is based on the DenseNet121 architecture, specifically designed for mammography image classification."
    },
    ('mammography', 'densenet', 'finetuned'): {
        'metrics': """Accuracy   : 0.9372
Precision  : 0.8145
Recall     : 0.6700
F1-score   : 0.7352
ROC-AUC    : 0.9630""",
        'report': """               precision    recall  f1-score   support

           0     0.9563    0.9792    0.9676      4380
           1     0.8145    0.6700    0.7352       650

    accuracy                         0.9372      5030
   macro avg     0.8854    0.8246    0.8514      5030
weighted avg     0.9380    0.9372    0.9363      5030""",
        'about': "This model is based on the DenseNet121 architecture with fine-tuning, specifically designed for mammography image classification."
    },
    # Ultrasound - ResNet
    ('ultrasound', 'resnet', 'initial'): {
        'metrics': """Accuracy   : 0.9897
Precision  : 0.9924
Recall     : 0.9865
F1-score   : 0.9895
ROC-AUC    : 0.9990""",
        'report': """               precision    recall  f1-score   support

           0     0.9870    0.9927    0.9898       686
           1     0.9924    0.9865    0.9895       667

    accuracy                         0.9897      1353
   macro avg     0.9897    0.9896    0.9896      1353
weighted avg     0.9897    0.9897    0.9897      1353""",
        'about': "This model is based on the ResNet50 architecture, specifically designed for ultrasound image classification."
    },
    ('ultrasound', 'resnet', 'finetuned'): {
        'metrics': """Accuracy   : 0.9889
Precision  : 0.9940
Recall     : 0.9835
F1-score   : 0.9887
ROC-AUC    : 0.9995""",
        'report': """               precision    recall  f1-score   support

           0     0.9840    0.9942    0.9891       686
           1     0.9940    0.9835    0.9887       667

    accuracy                         0.9889      1353
   macro avg     0.9890    0.9888    0.9889      1353
weighted avg     0.9889    0.9889    0.9889      1353""",
        'about': "This model is based on the ResNet50 architecture with fine-tuning, specifically designed for ultrasound image classification."
    },
    # Ultrasound - VGG
    ('ultrasound', 'vgg', 'initial'): {
        'metrics': """Accuracy   : 0.9534
Precision  : 0.9594
Recall     : 0.9445
F1-score   : 0.9519
ROC-AUC    : 0.9921""",
        'report': """               precision    recall  f1-score   support

           0     0.9479    0.9620    0.9549       686
           1     0.9594    0.9445    0.9519       667

    accuracy                         0.9534      1353
   macro avg     0.9536    0.9533    0.9534      1353
weighted avg     0.9535    0.9534    0.9534      1353""",
        'about': "This model is based on the VGG16 architecture, specifically designed for ultrasound image classification."
    },
    ('ultrasound', 'vgg', 'finetuned'): {
        'metrics': """Accuracy   : 0.9867
Precision  : 0.9910
Recall     : 0.9820
F1-score   : 0.9865
ROC-AUC    : 0.9993""",
        'report': """               precision    recall  f1-score   support

           0     0.9826    0.9913    0.9869       686
           1     0.9910    0.9820    0.9865       667

    accuracy                         0.9867      1353
   macro avg     0.9868    0.9866    0.9867      1353
weighted avg     0.9867    0.9867    0.9867      1353""",
        'about': "This model is based on the VGG16 architecture with fine-tuning, specifically designed for ultrasound image classification."
    },
    # Ultrasound - DenseNet
    ('ultrasound', 'densenet', 'initial'): {
        'metrics': """Accuracy   : 0.9897
Precision  : 0.9939
Recall     : 0.9850
F1-score   : 0.9895
ROC-AUC    : 0.9996""",
        'report': """               precision    recall  f1-score   support

           0     0.9855    0.9942    0.9898       686
           1     0.9939    0.9850    0.9895       667

    accuracy                         0.9897      1353
   macro avg     0.9897    0.9896    0.9896      1353
weighted avg     0.9897    0.9897    0.9897      1353""",
        'about': "This model is based on the DenseNet121 architecture, specifically designed for ultrasound image classification."
    },
    ('ultrasound', 'densenet', 'finetuned'): {
        'metrics': """Accuracy   : 0.9882
Precision  : 0.9880
Recall     : 0.9880
F1-score   : 0.9880
ROC-AUC    : 0.9994""",
        'report': """               precision    recall  f1-score   support

           0     0.9884    0.9884    0.9884       686
           1     0.9880    0.9880    0.9880       667

    accuracy                         0.9882      1353
   macro avg     0.9882    0.9882    0.9882      1353
weighted avg     0.9882    0.9882    0.9882      1353""",
        'about': "This model is based on the DenseNet121 architecture with fine-tuning, specifically designed for ultrasound image classification."
    },
    # Histopathology - ResNet
    ('histopathology', 'resnet', 'initial'): {
        'metrics': """Accuracy   : 0.8990
Precision  : 0.9148
Recall     : 0.8753
F1-score   : 0.8946
ROC-AUC    : 0.9622""",
        'report': """               precision    recall  f1-score   support

           0     0.8839    0.9223    0.9027       952
           1     0.9148    0.8753    0.8946       915

    accuracy                         0.8990      1867
   macro avg     0.8994    0.8988    0.8987      1867
weighted avg     0.8991    0.8990    0.8987      1867""",
        'about': "This model is based on the ResNet50 architecture, specifically designed for histopathology image classification."
    },
    ('histopathology', 'resnet', 'finetuned'): {
        'metrics': """Accuracy   : 0.8712
Precision  : 0.8793
Recall     : 0.8546
F1-score   : 0.8668
ROC-AUC    : 0.9445""",
        'report': """               precision    recall  f1-score   support

           0     0.8637    0.8866    0.8750       952
           1     0.8793    0.8546    0.8668       915

    accuracy                         0.8712      1867
   macro avg     0.8715    0.8706    0.8709      1867
weighted avg     0.8713    0.8712    0.8710      1867""",
        'about': "This model is based on the ResNet50 architecture with fine-tuning, specifically designed for histopathology image classification."
    },
    # Histopathology - VGG
    ('histopathology', 'vgg', 'initial'): {
        'metrics': """Accuracy   : 0.8912
Precision  : 0.8989
Recall     : 0.8765
F1-score   : 0.8876
ROC-AUC    : 0.9571""",
        'report': """               precision    recall  f1-score   support

           0     0.8840    0.9055    0.8946       952
           1     0.8989    0.8765    0.8876       915

    accuracy                         0.8912      1867
   macro avg     0.8914    0.8910    0.8911      1867
weighted avg     0.8913    0.8912    0.8911      1867""",
        'about': "This model is based on the VGG16 architecture, specifically designed for histopathology image classification."
    },
    ('histopathology', 'vgg', 'finetuned'): {
        'metrics': """Accuracy   : 0.8571
Precision  : 0.8628
Recall     : 0.8437
F1-score   : 0.8532
ROC-AUC    : 0.9324""",
        'report': """               precision    recall  f1-score   support

           0     0.8518    0.8697    0.8607       952
           1     0.8628    0.8437    0.8532       915

    accuracy                         0.8571      1867
   macro avg     0.8573    0.8567    0.8569      1867
weighted avg     0.8572    0.8571    0.8570      1867""",
        'about': "This model is based on the VGG16 architecture with fine-tuning, specifically designed for histopathology image classification."
    },
    # Histopathology - DenseNet
    ('histopathology', 'densenet', 'initial'): {
        'metrics': """Accuracy   : 0.8910
Precision  : 0.9089
Recall     : 0.8656
F1-score   : 0.8867
ROC-AUC    : 0.9566""",
        'report': """               precision    recall  f1-score   support

           0     0.8746    0.9160    0.8948       952
           1     0.9089    0.8656    0.8867       915

    accuracy                         0.8910      1867
   macro avg     0.8918    0.8908    0.8908      1867
weighted avg     0.8914    0.8910    0.8909      1867""",
        'about': "This model is based on the DenseNet121 architecture, specifically designed for histopathology image classification."
    },
    ('histopathology', 'densenet', 'finetuned'): {
        'metrics': """Accuracy   : 0.8481
Precision  : 0.8609
Recall     : 0.8240
F1-score   : 0.8420
ROC-AUC    : 0.9226""",
        'report': """               precision    recall  f1-score   support

           0     0.8365    0.8718    0.8538       952
           1     0.8609    0.8240    0.8420       915

    accuracy                         0.8481      1867
   macro avg     0.8487    0.8479    0.8479      1867
weighted avg     0.8485    0.8481    0.8480      1867""",
        'about': "This model is based on the DenseNet121 architecture with fine-tuning, specifically designed for histopathology image classification."
    },
}


def get_model_metrics(modality, architecture, training_type):
    """Return metrics data for a specific model."""
    key = (modality, architecture, training_type)
    return MODEL_METRICS.get(key, {})
