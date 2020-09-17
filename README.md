# Ident_Etiquetas
Identificador de Etiquetas

#Prerequisites

Python 3.6 or higher

### for Linux 

```
pip install torch torchvision
pip install easyocr
```

### for Windows
```
pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install easyocr
```

### Execução
No código teria que indicar o IP da camera na variável url
ou  modificação do videoCapture para utilizar o webcam `cv2.VideoCapture(0)`

```
python Etiquetas_detecet_ocr.py
```
