<!--
This README describes the package. If you publish this package to pub.dev,
this README's contents appear on the landing page for your package.

For information about how to write a good package README, see the guide for
[writing package pages](https://dart.dev/tools/pub/writing-package-pages).

For general information about developing packages, see the Dart guide for
[creating packages](https://dart.dev/guides/libraries/create-packages)
and the Flutter guide for
[developing packages and plugins](https://flutter.dev/to/develop-packages).
-->
# TFLite Vision

Biblioteca Flutter/Dart para orquestrar modelos TFLite de segmentação e
classificação (pré-processamento, carregamento de modelos e helpers para uso
em apps Flutter).

Este pacote reúne utilitários para:

- Carregar modelos TFLite (segmentação e classificação);
- Pré-processar imagens (redimensionamento e normalização para tensores);
- Executar inferência de segmentação e, em seguida, classificação sobre a imagem segmentada;
- Trabalhar com assets de modelos e labels embutidos no pacote.

## Recursos

- AppState: gestor de estado baseado em `ChangeNotifier` que carrega os modelos, armazena imagens originais segmentadas e expõe métodos para selecionar imagem e executar inferências.
- PreProcessing: conversão entre `Uint8List` / `ui.Image` e tensores 4D usados pelos modelos (normalização 0.0–1.0 por padrão).
- Model wrappers (em `core/models.dart`): adaptadores para carregar modelos TFLite e interpretar saídas (retornam `SegmentationResult` e `ClassificationResult`).
- Utils & Results: helpers para extrair índices máximos, mapear labels e estruturas de resultado.

## Instalação

Opções para consumir o pacote em outro projeto:

- Via pub.dev (após publicação):

Adicione no `pubspec.yaml` do app consumidor:

```yaml
dependencies:
    tflite_vision: ^0.0.1
```

- Via Git (sem publicar):

```yaml
dependencies:
    tflite_vision:
        git:
            url: https://github.com/SEU-REPO/tflite_vision.git
            ref: main
```

- Via path (desenvolvimento local):

```yaml
dependencies:
    tflite_vision:
        path: ../tflite_vision
```

Após adicionar, rode:

```bash
flutter pub get
```

## Assets (modelos e labels)

O pacote inclui (ou deve incluir) os arquivos de modelo e labels em
`assets/` (por exemplo `assets/segment_float16.tflite`,
`assets/classify_float16.tflite`, `assets/segment_labels.txt`,
`assets/classify_labels.txt`). Certifique-se de declarar esses assets em
`pubspec.yaml` do pacote para que consumidores possam acessá‑los.

Quando o pacote é usado por um app Flutter, os assets do pacote ficam
disponíveis no bundle do app com o prefixo `packages/<packageName>/...`.

Exemplo de leitura de asset do pacote:

```dart
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;

final data = await rootBundle.load('packages/tflite_vision/assets/segment_float16.tflite');
final bytes = data.buffer.asUint8List();
// use 'bytes' para carregar o modelo em tflite_flutter
```

## Uso (exemplos)

Importe a API pública:

```dart
import 'package:tflite_vision/tflite_vision.dart';
```

Criar o gestor de estado com caminhos personalizados para modelos/labels:

```dart
final appState = AppState(
    segmentModelPath: 'packages/tflite_vision/assets/custom_segment.tflite',
    segmentLabelsPath: 'packages/tflite_vision/assets/segment_labels.txt',
    classifyModelPath: 'packages/tflite_vision/assets/classify_float16.tflite',
    classifyLabelsPath: 'packages/tflite_vision/assets/classify_labels.txt',
);
```

Selecionar uma imagem e executar inferência:

```dart
// imageBytes é Uint8List (por ex. lido de ImagePicker ou de um asset)
await appState.selectImage(imageBytes);
await appState.runInference();

// resultados:
final segmentation = appState.segmentationResult; // SegmentationResult?
final classification = appState.classificationResult; // ClassificationResult?
final processed = appState.processedImage; // Uint8List da imagem segmentada
```

Notas de uso:

- `AppState.runInference()` primeiro executa o modelo de segmentação sobre a imagem original e, em seguida, executa o modelo de classificação sobre a imagem segmentada.
- Os campos `segmentModelPath`, `segmentLabelsPath`, `classifyModelPath` e `classifyLabelsPath` são configuráveis — podem ser passados no construtor ou alterados em tempo de execução via `updateModelPaths(...)`.

## API rápida (principais símbolos)

- `AppState` — gestor central. Principais métodos:
  - `selectImage(Uint8List imageBytes)` — armazena a imagem original.
  - `runInference()` — executa segmentação e classificação (assíncrono).
  - `updateModelPaths(...)` — troca paths e recarrega os modelos.
  - getters: `isLoading`, `areModelsLoaded`, `classificationResult`,  `segmentationResult`, `processedImage`.

- `PreProcessing` — utilitários estáticos:
  - `fromUiImage(ui.Image)` — converte `ui.Image` para `Uint8List`.
  - `imageToTensor(Uint8List, int width, int height)` — retorna tensor 4D normalizado (0.0..1.0) compatível com modelos que esperam floats.
  - `tensorToImage(...)` e `tensorToImageBytes(...)` — inverter o tensor para imagem PNG.

- Model wrappers (`TFLiteSegmentModel`, `TFLiteClassifyModel`) — métodos:
  - `loadModel({required modelPath, required labelsPath})` — carrega modelo/labels (suporta paths/em bytes conforme implementado internamente).
  - `predict(...)` — executa inferência e retorna estruturas de resultado (`SegmentationResult`, `ClassificationResult`).

## Contrato esperado dos modelos

- Input shapes: `TFLite*Model.inputShape` é usado para redimensionar a imagem antes da inferência. Confirme o ordenamento (por exemplo, `[1, height, width, channels]`), pois o pré-processamento redimensiona conforme `width` e `height` passados.
- Normalização: a implementação atual normaliza pixels para 0.0–1.0 (float). Verifique se seus modelos esperam uint8 (0–255) ou float e ajuste o pré-processamento caso necessário.
- Channels: os utilitários trabalham com RGB (3 canais). Se seus modelos esperam RGBA ou grayscale, adapte `PreProcessing`.

## Boas práticas para empacotar e publicar

1. Atualize `pubspec.yaml` com `name`, `version`, `description`, `homepage` e `authors`.
2. Declare os assets (`flutter.assets:`) em `pubspec.yaml` do pacote.
3. Inclua `README.md`, `CHANGELOG.md` e `LICENSE` completos.
4. Teste localmente:

```bash
flutter pub get
flutter analyze
flutter test
flutter pub publish --dry-run
```

Publique em `pub.dev` ou forneça acesso via `git`/`path` no `pubspec.yaml` do projeto consumidor.

## Exemplo mínimo (sugestão)

Crie a pasta `example/` com um app Flutter que demonstra:

- Carregar modelos do pacote;
- Selecionar imagem (camera/gallery);
- Executar `runInference()` e mostrar `processedImage` + `classification`.

## Contribuindo

1. Abra uma issue descrevendo o problema/feature.
2. Faça um fork, crie uma branch e envie um pull request com testes e descrições claras.

## Licença

Este repositório inclui a `LICENSE` na raiz; verifique-a antes de usar.
