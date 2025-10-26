/// Biblioteca para definir classes de resultados de inferência de modelos.
/// Contém classes para representar resultados de classificação e segmentação.
/// 
/// Classes:
/// - Result: Classe abstrata base para resultados de inferência.
/// - ClassificationResult: Classe para resultados de classificação.
/// - SegmentationResult: Classe para resultados de segmentação.
library;

import 'dart:typed_data';
import 'package:image/image.dart' as img;

/// Classe Abstrata para representar o resultado de uma inferência de modelo.
/// 
/// Properties:
/// - confidence: Nível de confiança do resultado.
/// - timestamp: Data e hora em que o resultado foi gerado.
/// - originalImage: Imagem original que foi processada.
abstract class Result {
  final double confidence;
  final DateTime timestamp;
  final Uint8List originalImage;
  
  const Result({
    required this.confidence,
    required this.timestamp,
    required this.originalImage,
  });
}

/// Classe para representar o resultado de uma classificação.
/// 
/// Properties:
/// - label: Rótulo previsto pelo modelo.
/// - allProbabilities: (Opcional) Lista de probabilidades para todas as classes.
/// - confidence: Nível de confiança do resultado.
/// - timestamp: Data e hora em que o resultado foi gerado.
/// - originalImage: Imagem original que foi processada.
class ClassificationResult extends Result {
  final String label;
  final List<double>? allProbabilities;
  
  const ClassificationResult({
    required super.confidence,
    required super.timestamp,
    required super.originalImage,
    required this.label,
    this.allProbabilities,
  });
  
  @override
  String toString() {
    return 'ClassificationResult(label: $label, confidence: $confidence)';
  }
}

/// Classe para representar o resultado de uma segmentação.
/// 
/// Properties:
/// - segmentedImage: (Opcional) Imagem segmentada resultante.
/// - binaryMask: (Opcional) Máscara binária da segmentação.
/// - maskCoefficients: (Opcional) Coeficientes associados à máscara.
/// - bestSegmentationIndex: (Opcional) Índice da melhor segmentação.
/// - metadata: (Opcional) Metadados adicionais relacionados ao resultado.
/// - confidence: Nível de confiança do resultado.
/// - timestamp: Data e hora em que o resultado foi gerado.
/// - originalImage: Imagem original que foi processada.
class SegmentationResult extends Result {
  final Uint8List? segmentedImage;
  final img.Image? binaryMask;
  final List<double>? maskCoefficients;
  final int? bestSegmentationIndex;
  final Map<String, dynamic>? metadata;
  
  const SegmentationResult({
    required super.confidence,
    required super.timestamp,
    required super.originalImage,
    this.segmentedImage,
    this.binaryMask,
    this.maskCoefficients,
    this.bestSegmentationIndex,
    this.metadata,
  });
  
  @override
  String toString() {
    return 'SegmentationResult(confidence: $confidence, hasSegmentedImage: ${segmentedImage != null})';
  }
}