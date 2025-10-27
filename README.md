# TFLite Vision

Biblioteca Flutter/Dart para orquestrar modelos TFLite de segmentação e classificação, com utilitários para pré-processamento, carregamento de modelos, execução de inferência e helpers para uso em apps Flutter.

**Importante:** Este pacote espera que o usuário (app consumidor) forneça os arquivos de modelo (.tflite) e labels (.txt) nos assets do próprio app. O pacote não inclui modelos/labels por padrão.

Principais recursos:

- Carregamento e execução de modelos TFLite (segmentação e/ou classificação)
- Pré-processamento de imagens (redimensionamento, normalização, conversão para tensores)
- Helpers para manipulação de resultados e assets
- Pipeline completo via `AppState` (segmentação + classificação em sequência)

> **Dica:** Os arquivos de modelo e labels devem estar declarados na seção `assets:` do `pubspec.yaml` do app consumidor.

## Como fornecer os modelos e labels

Você deve adicionar os arquivos `.tflite` e `.txt` (labels) nos assets do seu app Flutter. Exemplo de estrutura:

```bash
assets/
  segment_float16.tflite
  classify_float16.tflite
  segment_labels.txt
  classify_labels.txt
```

No seu `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/segment_float16.tflite
    - assets/classify_float16.tflite
    - assets/segment_labels.txt
    - assets/classify_labels.txt
```

Para carregar um asset no seu app:

```dart
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;

final data = await rootBundle.load('assets/segment_float16.tflite');
final bytes = data.buffer.asUint8List();
// use 'bytes' para carregar o modelo em tflite_flutter
```

### 1. Classificação isolada

```dart
import 'package:tflite_vision/tflite_vision.dart';
import 'package:flutter/services.dart' show rootBundle;

// Carregar modelo e labels do assets do app
final classifyModel = await TFLiteClassifyModel.loadModel(
  modelPath: 'assets/classify_float16.tflite',
  labelsPath: 'assets/classify_labels.txt',
);

// Carregar imagem (exemplo: asset)
final imageBytes = (await rootBundle.load('assets/test_image.jpg')).buffer.asUint8List();

// Pré-processar imagem
final inputTensor = await PreProcessing.imageToTensor(
  imageBytes,
  classifyModel.inputShape[2],
  classifyModel.inputShape[1],
);

// Executar inferência
final result = await classifyModel.predict(inputTensor, imageBytes);
print('Classe: \\${result.label}, confiança: \\${result.confidence}');
```

### 2. Segmentação isolada

```dart
import 'package:tflite_vision/tflite_vision.dart';
import 'package:flutter/services.dart' show rootBundle;

final segmentModel = await TFLiteSegmentModel.loadModel(
  modelPath: 'assets/segment_float16.tflite',
  labelsPath: 'assets/segment_labels.txt',
);

final imageBytes = (await rootBundle.load('assets/test_image.jpg')).buffer.asUint8List();

final inputTensor = await PreProcessing.imageToTensor(
  imageBytes,
  segmentModel.inputShape[2],
  segmentModel.inputShape[1],
);

final result = await segmentModel.predict(inputTensor, imageBytes);
print('Segmentação: \\${result.label}, máscara: \\${result.mask?.length} bytes');
```

### 3. Pipeline completo com AppState (segmentação + classificação)

```dart
import 'package:tflite_vision/tflite_vision.dart';
import 'package:flutter/services.dart' show rootBundle;

final appState = AppState(
  segmentModelPath: 'assets/segment_float16.tflite',
  segmentLabelsPath: 'assets/segment_labels.txt',
  classifyModelPath: 'assets/classify_float16.tflite',
  classifyLabelsPath: 'assets/classify_labels.txt',
);

final imageBytes = (await rootBundle.load('assets/test_image.jpg')).buffer.asUint8List();
await appState.selectImage(imageBytes);
await appState.runInference();

final segmentation = appState.segmentationResult;
final classification = appState.classificationResult;
final processed = appState.processedImage;

print('Segmentação: \\${segmentation?.label}');
print('Classificação: \\${classification?.label} (confiança: \\${classification?.confidence})');
```

> **Nota:** Os exemplos assumem que os assets de modelo/labels estão corretamente declarados no `pubspec.yaml` do app consumidor. Ajuste os paths conforme necessário.
