/// Biblioteca de modelos para o pacote `tflite_vision`.
/// Contém classes para modelos de segmentação e classificação TFLite.
/// Classes:
/// - TFLiteSegmentModel: Modelo de segmentação TFLite.
/// - TFLiteClassifyModel: Modelo de classificação TFLite.
library;

import 'package:flutter/foundation.dart';
import './results.dart';
import './utils.dart';
import './base.dart';

/// Classe que estende [TFLiteModel] para modelos de segmentação.
/// 
/// Métodos:
/// - loadModel: Carrega o modelo de segmentação TFLite e os rótulos a partir dos caminhos fornecidos.
/// - predict: Realiza a predição de segmentação em uma imagem de entrada.
/// - forward: Executa a inferência do modelo com o tensor de entrada.
/// - getResult: Processa as saídas do modelo para gerar o resultado de segmentação
class TFLiteSegmentModel extends TFLiteModel {
  TFLiteSegmentModel({
    required super.mainInterpreter,
    required super.isolateInterpreter,
    required super.inputShape,
    required super.outputShapes,
    required super.labels
  });

  /// Carrega o modelo TFLite e os rótulos a partir dos caminhos fornecidos.
  /// Retorna uma instância de [TFLiteSegmentModel] pronta para uso.
  static Future<TFLiteSegmentModel> loadModel({
    required String modelPath,
    required String labelsPath,
  }) async {
    final baseModel = await TFLiteModel.loadModel(
      modelPath: modelPath,
      labelsPath: labelsPath,
    );

    return TFLiteSegmentModel(
      mainInterpreter: baseModel.mainInterpreter,
      isolateInterpreter: baseModel.isolateInterpreter,
      inputShape: baseModel.inputShape,
      outputShapes: baseModel.outputShapes,
      labels: baseModel.labels,
    );
  }

  /// Realiza a predição de segmentação em uma imagem de entrada.
  /// Retorna um [SegmentationResult] contendo os resultados da segmentação.
  Future<SegmentationResult> predict(
    List<List<List<List<double>>>> inputTensor,
    Uint8List originalImageBytes,
  ) async {
    final outputs = await forward(inputTensor, originalImageBytes);
    return getResult(outputs, originalImageBytes);
  }

  /// Executa a inferência do modelo com o tensor de entrada.
  /// Retorna um mapa contendo as saídas do modelo.
  Future<Map<int, Object>> forward(
    List<List<List<List<double>>>> inputTensor,
    Uint8List originalImageBytes,
  ) async {
    try {
      validateOutputShapes(outputShapes, 2);
      final outputs = createOutputBuffers(outputShapes);
      await isolateInterpreter.runForMultipleInputs([inputTensor], outputs);

      return outputs;
      
    } catch (error) {
      rethrow;
    }
  }

  /// Processa as saídas do modelo para gerar o resultado de segmentação
  /// Retorna um [SegmentationResult] contendo os resultados da segmentação.
  SegmentationResult getResult (
    Map<int, Object> outputs,
    Uint8List originalImageBytes,
  ) {
    final segmentationThreshold = 0.5; // Threshold de confiança
    final firstCoeffIndex = 5;  // Índice da primeira coluna dos coeficientes da máscara
    double confidence;
    
    final segmentations = (outputs[0] as List<List<List<double>>>)[0];
    final maskPrototypes = (outputs[1] as List<List<List<List<double>>>>)[0];
    
    final bestSegmentationIndex = getBestSegmentationIndex(segmentations, segmentationThreshold);
    if (bestSegmentationIndex == -1) {
      throw Exception("No found segmentations.");
    }

    final maskCoeffs = extractMaskCoefficients(
      segmentations, 
      bestSegmentationIndex,
      firstCoeffIndex
    );

    final binaryMask = buildBinaryMask(maskPrototypes, maskCoeffs);
    final originalImage = decodeOriginalImage(originalImageBytes);
    final resizedMask = resizeMask(binaryMask, originalImage.width, originalImage.height);
    final maskedImage = applyMaskToImage(originalImage, resizedMask);
    final segmentedImageBytes = encodeImageToPng(maskedImage);

    confidence = bestSegmentationIndex < segmentations[0].length ? segmentations[4][bestSegmentationIndex] : 0.0;
    
    return SegmentationResult(
      originalImage: originalImageBytes,
      segmentedImage: segmentedImageBytes,
      binaryMask: binaryMask,
      maskCoefficients: maskCoeffs,
      bestSegmentationIndex: bestSegmentationIndex,
      confidence: confidence,
      timestamp: DateTime.now(),
      metadata: {
        'threshold': segmentationThreshold,
        'firstCoeffIndex': firstCoeffIndex,
        'originalImageSize': '${originalImage.width}x${originalImage.height}',
      },
    );
  }
}

/// Classe que estende [TFLiteModel] para modelos de classificação.
/// 
/// Métodos:
/// - loadModel: Carrega o modelo de classificação TFLite e os rótulos a partir dos caminhos fornecidos.
/// - predict: Realiza a predição de classificação em uma imagem de entrada.
/// - forward: Executa a inferência do modelo com o tensor de entrada.
/// - getResult: Processa as saídas do modelo para gerar o resultado de classificação
class TFLiteClassifyModel extends TFLiteModel {
  TFLiteClassifyModel(
      {required super.mainInterpreter,
      required super.isolateInterpreter,
      required super.inputShape,
      required super.outputShapes,
      required super.labels});

  /// Carrega o modelo TFLite e os rótulos a partir dos caminhos fornecidos.
  /// Retorna uma instância de [TFLiteClassifyModel] pronta para uso
  static Future<TFLiteClassifyModel> loadModel({
    required String modelPath,
    required String labelsPath,
  }) async {
    final baseModel = await TFLiteModel.loadModel(
      modelPath: modelPath,
      labelsPath: labelsPath,
    );

    return TFLiteClassifyModel(
      mainInterpreter: baseModel.mainInterpreter,
      isolateInterpreter: baseModel.isolateInterpreter,
      inputShape: baseModel.inputShape,
      outputShapes: baseModel.outputShapes,
      labels: baseModel.labels,
    );
  }
  /// Realiza a predição de classificação em uma imagem de entrada.
  /// Retorna um [ClassificationResult] contendo os resultados da classificação.
  Future<ClassificationResult> predict(
    List<List<List<List<double>>>> inputTensor,
    Uint8List originalImageBytes,
  ) async {
    final outputs = await forward(inputTensor, originalImageBytes);
    return getResult(outputs, originalImageBytes);
  }

  /// Executa a inferência do modelo com o tensor de entrada.
  /// Retorna um mapa contendo as saídas do modelo.
  Future<Map<int, Object>> forward(
    List<List<List<List<double>>>> inputTensor,
    Uint8List originalImageBytes,
  ) async {
    try {
      final output = createOutputBuffers(outputShapes);
      await isolateInterpreter.runForMultipleInputs([inputTensor], output);
      return output;

    } catch (error) {
      rethrow;
    }
  }

  /// Processa as saídas do modelo para gerar o resultado de classificação
  /// Retorna um [ClassificationResult] contendo os resultados da classificação.
  ClassificationResult getResult (
    Map<int, Object> outputs,
    Uint8List originalImageBytes,
  ) {
    final probabilities = (outputs[0] as List<List<double>>)[0];
    final (maxIndex, maxProb) = getMaxIndexAndProb(probabilities);
    
    return ClassificationResult(
      originalImage: originalImageBytes,
      label: labels[maxIndex],
      confidence: maxProb,
      allProbabilities: probabilities,
      timestamp: DateTime.now(),
    );
  }
}